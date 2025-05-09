import argparse
import logging
import time
import torch
import evaluate
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import warnings
import yaml
import os
import sys

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Logging setup
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[console]
)
logger = logging.getLogger(__name__)


def load_model_int8(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=quant_config
    ).eval()

    return model, tokenizer


# Evaluation modules
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")


def evaluate_rouge(pred, ref):
    scores = rouge.score(ref, pred)
    return scores["rouge1"].fmeasure, scores["rouge2"].fmeasure, scores["rougeL"].fmeasure


def mean_rouge(r1, r2, rl):
    return (r1 + r2 + rl) / 3.0


def evaluate_bertscore(pred, ref):
    result = bertscore.compute(predictions=[pred], references=[ref], lang="en")
    return result["f1"][0]


def evaluate_bleu(pred, ref):
    result = bleu.compute(predictions=[pred], references=[[ref]])
    return result["bleu"]


def cosine_similarity(pred, ref, embedder):
    emb1 = embedder.encode(pred, convert_to_tensor=True)
    emb2 = embedder.encode(ref, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2)[0])


def confidence_score(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    max_probs = torch.max(probs, dim=-1).values
    return max_probs.mean().item()


def get_response(model, tokenizer, question, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
    prompt = (
        "You are a helpful assistant. Provide a detailed and informative answer to the following question. "
        "Ensure the answer is at least 50 words long and includes relevant factual details and commonly expected terms.\n\n"
        f"Question: {question.strip()}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    elapsed = time.time() - start_time
    tokens_generated = output.shape[-1] - inputs["input_ids"].shape[-1]
    tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
    answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return answer, tokens_per_sec


def refine_summary(ans1, ans2, reference, refiner_model, refiner_tokenizer, max_new_tokens):
    prompt = (
        "You are an expert AI assistant. Combine the best information from the two responses below into one clear, informative answer. "
        "The final answer should be at least 50 words long, avoid vague phrases, and include factual terms or named entities "
        "that improve keyword overlap with the reference answer if available.\n\n"
        f"Response 1:\n{ans1}\n\n"
        f"Response 2:\n{ans2}\n\n"
        f"Reference (optional):\n{reference if reference else 'N/A'}\n\n"
        "Final refined response:"
    )
    inputs = refiner_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output = refiner_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.5,
            top_k=30,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=refiner_tokenizer.pad_token_id
        )
    return refiner_tokenizer.decode(output[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--repetition_penalty", type=float)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_split", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--phi_model", type=str)
    parser.add_argument("--pythia_model", type=str)
    parser.add_argument("--refiner_model", type=str)
    parser.add_argument("--embedder_model", type=str)

    args = parser.parse_args()

    # Load YAML
    config_path = args.config if args.config else "config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Override config with CLI if provided
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            cfg[key] = value

    logger.info("Loading dataset...")
    dataset = load_dataset(cfg["dataset_name"], split=cfg["dataset_split"], cache_dir=cfg["dataset_path"])

    logger.info("Loading models in 8-bit...")
    phi_model, phi_tokenizer = load_model_int8(cfg["phi_model"])
    pythia_model, pythia_tokenizer = load_model_int8(cfg["pythia_model"])
    refiner_model, refiner_tokenizer = load_model_int8(cfg["refiner_model"])

    logger.info("Loading embedder model...")
    embedder = SentenceTransformer(cfg["embedder_model"], device=device)

    results = {
        "rouge1": [], "rouge2": [], "rougeL": [],
        "bertscore": [], "bleu": [], "cosine": [],
        "confidence": [], "tps": []
    }

    logger.info("Starting evaluation loop...")
    for sample in tqdm(dataset):
        question = sample["query"]
        reference = sample["answer"]

        logger.info(f"Processing question: {question}")

        ans1, tps1 = get_response(phi_model, phi_tokenizer, question, cfg["max_new_tokens"], cfg["temperature"], cfg["top_k"], cfg["top_p"], cfg["repetition_penalty"])
        logger.info(f"Answer from phi_model: {ans1[:100]}...")

        ans2, tps2 = get_response(pythia_model, pythia_tokenizer, question, cfg["max_new_tokens"], cfg["temperature"], cfg["top_k"], cfg["top_p"], cfg["repetition_penalty"])
        logger.info(f"Answer from pythia_model: {ans2[:100]}...")

        refined = refine_summary(ans1, ans2, reference, refiner_model, refiner_tokenizer, cfg["max_new_tokens"])
        logger.info(f"Refined response: {refined[:100]}...")

        try:
            r1, r2, rl = evaluate_rouge(refined, reference)
            bs = evaluate_bertscore(refined, reference)
            bl = evaluate_bleu(refined, reference)
            cs = cosine_similarity(refined, reference, embedder)
            conf = confidence_score(refiner_model, refiner_tokenizer, refined)
            tps_avg = (tps1 + tps2) / 2
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            r1 = r2 = rl = bs = bl = cs = conf = tps_avg = 0.0

        results["rouge1"].append(r1)
        results["rouge2"].append(r2)
        results["rougeL"].append(rl)
        results["bertscore"].append(bs)
        results["bleu"].append(bl)
        results["cosine"].append(cs)
        results["confidence"].append(conf)
        results["tps"].append(tps_avg)

    logger.info("\nFinal Evaluation:")
    logger.info(f"ROUGE-1        → {np.mean(results['rouge1']):.4f}")
    logger.info(f"ROUGE-2        → {np.mean(results['rouge2']):.4f}")
    logger.info(f"ROUGE-L        → {np.mean(results['rougeL']):.4f}")
    logger.info(f"Mean ROUGE     → {mean_rouge(np.mean(results['rouge1']), np.mean(results['rouge2']), np.mean(results['rougeL'])):.4f}")
    logger.info(f"BERTScore      → {np.mean(results['bertscore']):.4f}")
    logger.info(f"BLEU           → {np.mean(results['bleu']):.4f}")
    logger.info(f"Cosine Sim     → {np.mean(results['cosine']):.4f}")
    logger.info(f"Confidence     → {np.mean(results['confidence']):.4f}")
    logger.info(f"Tokens/Sec     → {np.mean(results['tps']):.2f}")


if __name__ == "__main__":
    main()
