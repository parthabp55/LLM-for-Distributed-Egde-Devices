import argparse
import logging
import time
import torch
import evaluate
import numpy as np
import yaml
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import warnings
import os

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# Load int8 model
def load_int8_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=quant_config
    ).eval()

    return model, tokenizer

# Metric functions
def evaluate_rouge(pred, ref):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return scores["rouge1"].fmeasure, scores["rouge2"].fmeasure, scores["rougeL"].fmeasure

def evaluate_bertscore(pred, ref):
    bertscore = evaluate.load("bertscore")
    result = bertscore.compute(predictions=[pred], references=[ref], lang="en")
    return result["f1"][0]

def evaluate_bleu(pred, ref):
    bleu = evaluate.load("bleu")
    result = bleu.compute(predictions=[pred], references=[[ref]])
    return result["bleu"]

def cosine_similarity(pred, ref):
    emb1 = EMBEDDING_MODEL.encode(pred, convert_to_tensor=True)
    emb2 = EMBEDDING_MODEL.encode(ref, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2)[0])

def confidence_score(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    max_probs = torch.max(probs, dim=-1).values
    return max_probs.mean().item()

def tokens_per_second(start_time, num_tokens):
    return num_tokens / (time.time() - start_time)

def get_response(model, tokenizer, question, config):
    prompt = (
        "You are a helpful assistant. Provide a detailed and informative answer to the following question. "
        "Ensure the answer is at least 50 words long and includes relevant factual details and commonly expected terms.\n\n"
        f"Question: {question.strip()}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"],
            repetition_penalty=config["repetition_penalty"],
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    num_tokens = len(output[0])
    tps = tokens_per_second(start_time, num_tokens)
    return generated_text, tps

def mean_rouge(r1, r2, rl):
    return (r1 + r2 + rl) / 3.0

def evaluate_model(model_path, dataset, config):
    logger.info(f"\n--- Evaluating Model: {model_path} ---")
    model, tokenizer = load_int8_model(model_path)

    all_r1, all_r2, all_rl = [], [], []
    all_bs, all_bleu, all_confidence, all_similarity, all_tps = [], [], [], [], []

    for sample in tqdm(dataset):
        question = sample["query"]
        reference = sample["answer"]
        response, tps = get_response(model, tokenizer, question, config)

        try:
            r1, r2, rl = evaluate_rouge(response, reference)
            bs = evaluate_bertscore(response, reference)
            bleu = evaluate_bleu(response, reference)
            conf = confidence_score(model, tokenizer, response)
            sim = cosine_similarity(response, reference)
        except:
            r1 = r2 = rl = bs = bleu = conf = sim = 0.0

        all_r1.append(r1)
        all_r2.append(r2)
        all_rl.append(rl)
        all_bs.append(bs)
        all_bleu.append(bleu)
        all_confidence.append(conf)
        all_similarity.append(sim)
        all_tps.append(tps)

    logger.info(f"ROUGE-1         → {np.mean(all_r1):.4f}")
    logger.info(f"ROUGE-2         → {np.mean(all_r2):.4f}")
    logger.info(f"ROUGE-L         → {np.mean(all_rl):.4f}")
    logger.info(f"BERTScore       → {np.mean(all_bs):.4f}")
    logger.info(f"BLEU Score      → {np.mean(all_bleu):.4f}")
    logger.info(f"Confidence Score→ {np.mean(all_confidence):.4f}")
    logger.info(f"Similarity Score→ {np.mean(all_similarity):.4f}")
    logger.info(f"Tokens/sec      → {np.mean(all_tps):.4f}")
    logger.info(f" Mean ROUGE     → {mean_rouge(np.mean(all_r1), np.mean(all_r2), np.mean(all_rl)):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--models", nargs="+", help="List of models to evaluate (override YAML)")
    parser.add_argument("--dataset_name", type=str, help="Dataset name (override YAML)")
    parser.add_argument("--dataset_split", type=str, help="Dataset split (override YAML)")
    parser.add_argument("--max_new_tokens", type=int, help="Max new tokens")
    parser.add_argument("--temperature", type=float, help="Temperature")
    parser.add_argument("--top_k", type=int, help="Top-K sampling")
    parser.add_argument("--top_p", type=float, help="Top-P sampling")
    parser.add_argument("--repetition_penalty", type=float, help="Repetition penalty")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["models"] = args.models or config["models"]
    config["dataset_name"] = args.dataset_name or config["dataset_name"]
    config["dataset_split"] = args.dataset_split or config["dataset_split"]
    config["max_new_tokens"] = args.max_new_tokens or config["max_new_tokens"]
    config["temperature"] = args.temperature or config["temperature"]
    config["top_k"] = args.top_k or config["top_k"]
    config["top_p"] = args.top_p or config["top_p"]
    config["repetition_penalty"] = args.repetition_penalty or config["repetition_penalty"]

    dataset = load_dataset(config["dataset_name"], split=config["dataset_split"])
    base_model_path = "/home/research/Desktop/Models"

    for model_name in config["models"]:
        full_path = os.path.join(base_model_path, model_name)
        evaluate_model(full_path, dataset, config)

if __name__ == "__main__":
    main()
