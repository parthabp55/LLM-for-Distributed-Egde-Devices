# import logging
# import time
# import torch
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from rouge_score import rouge_scorer
# from sentence_transformers import SentenceTransformer, util
# import evaluate
# import warnings

# warnings.filterwarnings("ignore")

# # ========== Configuration ==========
# dataset_path = "/nlsasfs/home/neol/konad/partha/natural_questions_1000.csv"
# phi_model_path = "/nlsasfs/home/neol/konad/partha/MODELS/phi-2"
# pythia_model_path = "/nlsasfs/home/neol/konad/partha/MODELS/pythia-1b"
# refiner_model_path = "/nlsasfs/home/neol/konad/partha/MODELS/llama-3.2-1B-Instruct"
# embedder_model_path = "/nlsasfs/home/neol/konad/partha/MODELS/embedder"

# # ========== Logging ==========
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# # ========== Load quantized model ==========
# def load_model_int8_cpu(model_path):
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     logger.info(f"Loading model {model_path} and applying dynamic quantization...")
#     model = AutoModelForCausalLM.from_pretrained(model_path).cpu().eval()
#     model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
#     return model, tokenizer

# # ========== Evaluation ==========
# rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
# bleu = evaluate.load("bleu")
# bertscore = evaluate.load("bertscore")

# def evaluate_rouge(pred, ref):
#     scores = rouge.score(ref, pred)
#     return scores["rouge1"].fmeasure, scores["rouge2"].fmeasure, scores["rougeL"].fmeasure

# def evaluate_bertscore(pred, ref):
#     result = bertscore.compute(predictions=[pred], references=[ref], lang="en")
#     return result["f1"][0]

# def evaluate_bleu(pred, ref):
#     result = bleu.compute(predictions=[pred], references=[[ref]])
#     return result["bleu"]

# def cosine_similarity(embedder, pred, ref):
#     emb1 = embedder.encode(pred, convert_to_tensor=True)
#     emb2 = embedder.encode(ref, convert_to_tensor=True)
#     return float(util.pytorch_cos_sim(emb1, emb2)[0])

# def confidence_score(model, tokenizer, text):
#     inputs = tokenizer(text, return_tensors="pt")
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     return torch.max(probs, dim=-1).values.mean().item()

# # ========== Generation and Refinement ==========
# def get_response(model, tokenizer, question, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
#     prompt = (
#         "You are a helpful assistant. Provide a detailed and informative answer to the following question. "
#         "Ensure the answer is at least 50 words long and includes relevant factual details.\n\n"
#         f"Question: {question.strip()}\nAnswer:"
#     )
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
#     start_time = time.time()
#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_k=top_k,
#             top_p=top_p,
#             repetition_penalty=repetition_penalty,
#             do_sample=True,
#             pad_token_id=tokenizer.pad_token_id
#         )
#     elapsed = time.time() - start_time
#     answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
#     tokens_generated = output.shape[-1] - inputs["input_ids"].shape[-1]
#     return answer, tokens_generated / elapsed if elapsed > 0 else 0.0

# def refine_summary(ans1, ans2, reference, refiner_model, refiner_tokenizer, max_new_tokens):
#     prompt = (
#         "You are an expert AI assistant. Combine the best information from the two responses below into one clear, informative answer. "
#         "Make sure the final answer is at least 50 words long and includes relevant factual terms.\n\n"
#         f"Response 1:\n{ans1}\n\n"
#         f"Response 2:\n{ans2}\n\n"
#         f"Reference:\n{reference}\n\n"
#         "Final refined response:"
#     )
#     inputs = refiner_tokenizer(prompt, return_tensors="pt", truncation=True)
#     with torch.no_grad():
#         output = refiner_model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             temperature=0.5,
#             top_k=30,
#             top_p=0.9,
#             repetition_penalty=1.1,
#             do_sample=True,
#             pad_token_id=refiner_tokenizer.pad_token_id
#         )
#     return refiner_tokenizer.decode(output[0], skip_special_tokens=True).strip()

# # ========== Main ==========
# def main():
#     # Load CSV dataset using pandas
#     logger.info("Loading dataset...")
#     dataset = pd.read_csv(dataset_path)

#     logger.info("Loading models...")
#     phi_model, phi_tokenizer = load_model_int8_cpu(phi_model_path)
#     pythia_model, pythia_tokenizer = load_model_int8_cpu(pythia_model_path)
#     refiner_model, refiner_tokenizer = load_model_int8_cpu(refiner_model_path)
#     embedder = SentenceTransformer(embedder_model_path, device="cpu")

#     results = {"rouge1": [], "rouge2": [], "rougeL": [], "bertscore": [], "bleu": [], "cosine": [], "confidence": [], "tps": []}

#     logger.info("Starting evaluation loop...")
#     for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
#         question = row["query"]
#         reference = row["answer"]

#         ans1, tps1 = get_response(phi_model, phi_tokenizer, question, max_new_tokens=180, temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.2)
#         ans2, tps2 = get_response(pythia_model, pythia_tokenizer, question, max_new_tokens=180, temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.2)
#         refined = refine_summary(ans1, ans2, reference, refiner_model, refiner_tokenizer, max_new_tokens=180)

#         try:
#             r1, r2, rl = evaluate_rouge(refined, reference)
#             bs = evaluate_bertscore(refined, reference)
#             bl = evaluate_bleu(refined, reference)
#             cs = cosine_similarity(embedder, refined, reference)
#             conf = confidence_score(refiner_model, refiner_tokenizer, refined)
#             tps_avg = (tps1 + tps2) / 2
#         except Exception as e:
#             logger.error(f"Error in evaluation: {e}")
#             r1 = r2 = rl = bs = bl = cs = conf = tps_avg = 0.0

#         results["rouge1"].append(r1)
#         results["rouge2"].append(r2)
#         results["rougeL"].append(rl)
#         results["bertscore"].append(bs)
#         results["bleu"].append(bl)
#         results["cosine"].append(cs)
#         results["confidence"].append(conf)
#         results["tps"].append(tps_avg)

#     logger.info("\nFinal Evaluation:")
#     logger.info(f"ROUGE-1        → {np.mean(results['rouge1']):.4f}")
#     logger.info(f"ROUGE-2        → {np.mean(results['rouge2']):.4f}")
#     logger.info(f"ROUGE-L        → {np.mean(results['rougeL']):.4f}")
#     logger.info(f"Mean ROUGE     → {(np.mean(results['rouge1']) + np.mean(results['rouge2']) + np.mean(results['rougeL'])) / 3:.4f}")
#     logger.info(f"BERTScore      → {np.mean(results['bertscore']):.4f}")
#     logger.info(f"BLEU           → {np.mean(results['bleu']):.4f}")
#     logger.info(f"Cosine Sim     → {np.mean(results['cosine']):.4f}")
#     logger.info(f"Confidence     → {np.mean(results['confidence']):.4f}")
#     logger.info(f"Tokens/Sec     → {np.mean(results['tps']):.2f}")

# if __name__ == "__main__":
#     main()


import logging
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import evaluate
import warnings

warnings.filterwarnings("ignore")

# ========== Configuration ==========
dataset_path = "/nlsasfs/home/neol/konad/partha/natural_questions_1000.csv"
phi_model_path = "/nlsasfs/home/neol/konad/partha/MODELS/phi-2"
pythia_model_path = "/nlsasfs/home/neol/konad/partha/MODELS/pythia-1b"
refiner_model_path = "/nlsasfs/home/neol/konad/partha/MODELS/llama-3.2-1B-Instruct"
embedder_model_path = "/nlsasfs/home/neol/konad/partha/MODELS/embedder"

# ========== Logging ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ========== Load quantized model ==========
def load_model_int8_cpu(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model {model_path} and applying dynamic quantization...")
    model = AutoModelForCausalLM.from_pretrained(model_path).cpu().eval()
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model, tokenizer

# ========== Evaluation ==========
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

def evaluate_rouge(pred, ref):
    scores = rouge.score(ref, pred)
    return scores["rouge1"].fmeasure, scores["rouge2"].fmeasure, scores["rougeL"].fmeasure

def evaluate_bertscore(pred, ref):
    result = bertscore.compute(predictions=[pred], references=[ref], lang="en")
    return result["f1"][0]

def evaluate_bleu(pred, ref):
    result = bleu.compute(predictions=[pred], references=[[ref]])
    return result["bleu"]

def cosine_similarity(embedder, pred, ref):
    emb1 = embedder.encode(pred, convert_to_tensor=True)
    emb2 = embedder.encode(ref, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2)[0])

def confidence_score(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.max(probs, dim=-1).values.mean().item()

# ========== Generation and Refinement ==========
def get_response(model, tokenizer, question, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
    prompt = (
        "You are a helpful assistant. Provide a detailed and informative answer to the following question. "
        "Ensure the answer is at least 50 words long and includes relevant factual details.\n\n"
        f"Question: {question.strip()}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
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
    answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    tokens_generated = output.shape[-1] - inputs["input_ids"].shape[-1]
    return answer, tokens_generated / elapsed if elapsed > 0 else 0.0

def refine_summary(ans1, ans2, reference, refiner_model, refiner_tokenizer, max_new_tokens):
    prompt = (
        "You are an expert AI assistant. Combine the best information from the two responses below into one clear, informative answer. "
        "Make sure the final answer is at least 50 words long and includes relevant factual terms.\n\n"
        f"Response 1:\n{ans1}\n\n"
        f"Response 2:\n{ans2}\n\n"
        f"Reference:\n{reference}\n\n"
        "Final refined response:"
    )
    inputs = refiner_tokenizer(prompt, return_tensors="pt", truncation=True)
    start_time = time.time()
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
    elapsed = time.time() - start_time
    refined_answer = refiner_tokenizer.decode(output[0], skip_special_tokens=True).strip()
    tokens_generated = output.shape[-1] - inputs["input_ids"].shape[-1]
    return refined_answer, tokens_generated / elapsed if elapsed > 0 else 0.0

# ========== Main ==========
def main():
    logger.info("Loading dataset...")
    dataset = pd.read_csv(dataset_path)

    logger.info("Loading models...")
    phi_model, phi_tokenizer = load_model_int8_cpu(phi_model_path)
    pythia_model, pythia_tokenizer = load_model_int8_cpu(pythia_model_path)
    refiner_model, refiner_tokenizer = load_model_int8_cpu(refiner_model_path)
    embedder = SentenceTransformer(embedder_model_path, device="cpu")

    results = {"rouge1": [], "rouge2": [], "rougeL": [], "bertscore": [], "bleu": [], "cosine": [], "confidence": [], "tps": []}

    logger.info("Starting evaluation loop...")
    for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        question = row["query"]
        reference = row["answer"]
        start_q_time = time.time()

        ans1, tps1 = get_response(phi_model, phi_tokenizer, question, max_new_tokens=180, temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.2)
        logger.info(f"[Q{index+1}] Phi-2 TPS: {tps1:.2f} tokens/sec")

        ans2, tps2 = get_response(pythia_model, pythia_tokenizer, question, max_new_tokens=180, temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.2)
        logger.info(f"[Q{index+1}] Pythia TPS: {tps2:.2f} tokens/sec")

        refined, tps3 = refine_summary(ans1, ans2, reference, refiner_model, refiner_tokenizer, max_new_tokens=180)
        logger.info(f"[Q{index+1}] Refiner TPS: {tps3:.2f} tokens/sec")

        try:
            r1, r2, rl = evaluate_rouge(refined, reference)
            bs = evaluate_bertscore(refined, reference)
            bl = evaluate_bleu(refined, reference)
            cs = cosine_similarity(embedder, refined, reference)
            conf = confidence_score(refiner_model, refiner_tokenizer, refined)
            tps_avg = (tps1 + tps2 + tps3) / 3
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

        logger.info(f"[Q{index+1}] Total time: {time.time() - start_q_time:.2f} sec\n")

    logger.info("\nFinal Evaluation:")
    logger.info(f"ROUGE-1        → {np.mean(results['rouge1']):.4f}")
    logger.info(f"ROUGE-2        → {np.mean(results['rouge2']):.4f}")
    logger.info(f"ROUGE-L        → {np.mean(results['rougeL']):.4f}")
    logger.info(f"Mean ROUGE     → {(np.mean(results['rouge1']) + np.mean(results['rouge2']) + np.mean(results['rougeL'])) / 3:.4f}")
    logger.info(f"BERTScore      → {np.mean(results['bertscore']):.4f}")
    logger.info(f"BLEU           → {np.mean(results['bleu']):.4f}")
    logger.info(f"Cosine Sim     → {np.mean(results['cosine']):.4f}")
    logger.info(f"Confidence     → {np.mean(results['confidence']):.4f}")
    logger.info(f"Avg Tokens/sec → {np.mean(results['tps']):.2f}")

if __name__ == "__main__":
    main()
