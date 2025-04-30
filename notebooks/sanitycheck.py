import torch
from transformers import AutoTokenizer, AutoModel
import argparse

import torch.nn.functional as F

MODEL_CONFIGS = {
    "gpt2": {
        "model_name": "gpt2",
        "tokenizer_name": "gpt2",
    },
    "pythia1.4b": {
        "model_name": "EleutherAI/pythia-1.4b-v0",
        "tokenizer_name": "EleutherAI/pythia-1.4b-v0",
    },
    "gemma2b": {
        "model_name": "google/gemma-2-2b",
        "tokenizer_name": "google/gemma-2-2b",
    },
    "qwen2": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "tokenizer_name": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "bert-base-uncased": {
        "model_name": "bert-base-uncased",
        "tokenizer_name": "bert-base-uncased",
    },
    "bert-large-uncased": {
        "model_name": "bert-large-uncased",
        "tokenizer_name": "bert-large-uncased",
    },
    "distilbert-base-uncased": {
        "model_name": "distilbert-base-uncased",
        "tokenizer_name": "distilbert-base-uncased",
    },
}

def get_embedding(tokenizer, embeddings, word, method="sum"):
    if method == "tokenize":
        toks = tokenizer.tokenize(word, add_special_tokens=False)
        ids = tokenizer.convert_tokens_to_ids(toks)
        vecs = embeddings[ids]
        return vecs.mean(dim=0)
    else: # sum
        toks = tokenizer.tokenize(" " + word, add_special_tokens=False)
        ids = tokenizer.convert_tokens_to_ids(toks)
        vecs = embeddings[ids]
        return vecs.sum(dim=0)
    
def get_word_rank(tokenizer, embeddings, query_vec, word, method="sum"):
    emb_norm = F.normalize(embeddings, dim=1)
    q_norm = F.normalize(query_vec.unsqueeze(0), dim=1)
    sims = torch.mm(q_norm, emb_norm.t()).squeeze(0)

    if method == "tokenize":
        toks = tokenizer.tokenize(word, add_special_tokens=False)
        ids_for_rank = tokenizer.convert_tokens_to_ids(toks)
    else:  # sum
        toks = tokenizer.tokenize(" " + word, add_special_tokens=False)
        ids_for_rank = tokenizer.convert_tokens_to_ids(toks)
    
    sorted_idxs = torch.argsort(sims, descending=True)
    ranks = []
    for tid in ids_for_rank:
        pos = (sorted_idxs == tid).nonzero(as_tuple=True)[0]
        ranks.append(pos.item() + 1)
    return sum(ranks) / len(ranks)

def find_closest(tokenizer, embeddings, query_vec, top_k=5):
    emb_norm = F.normalize(embeddings, dim=1)
    q_norm = F.normalize(query_vec.unsqueeze(0), dim=1)
    sims = torch.mm(q_norm, emb_norm.t()).squeeze(0)
    vals, idxs = torch.topk(sims, k=top_k*10)
    results, seen = [], set()
    
    for score, idx in zip(vals.tolist(), idxs.tolist()):
        tok = tokenizer.decode([idx]).strip()
        # tok = tokenizer.convert_ids_to_tokens([idx])[0].strip()
        if not tok.isalpha() or tok in seen: # don't include byte-level tokens
            continue
        seen.add(tok)
        results.append((tok, score))
        if len(results) >= top_k:
            break
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Run embedding analogy tests.")
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()), required=True,
                        help="Model name to run the tests on.")
    args = parser.parse_args()

    tests = [
        ("king", "man", "woman", "queen"),
        ("man", "king", "queen", "woman"),
        ("walked", "walk", "jump", "jumped"),
        ("go", "went", "run", "ran"),
        ("sang", "sing", "ring", "rang"),
        ("sing", "sang", "rang", "ring"),
    ]

    model_key = args.model
    print(f"\n\n==================== {model_key} ====================")
    cfg = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
    model = AutoModel.from_pretrained(cfg["model_name"])
    embeddings = model.get_input_embeddings().weight.data

    for a, b, c, d in tests:
        print(f"\n=== Analogy ({a}-{b}+{c}) expecting {d} ===")
        for method in ("tokenize", "sum"):
            va = get_embedding(tokenizer, embeddings, a, method=method)
            vb = get_embedding(tokenizer, embeddings, b, method=method)
            vc = get_embedding(tokenizer, embeddings, c, method=method)
            query = va - vb + vc

            rank = get_word_rank(tokenizer, embeddings, query, d)
            print(f"\n method={method}: rank of '{d}' = {int(rank)}")
            for tok, sim in find_closest(tokenizer, embeddings, query, top_k=5):
                print(f"   {tok!r:<10} sim={sim:.4f}")
                # print(f"   {tok!r}  cos_sim={sim:.4f}")

    print("\n=== E('ed') + E('jump') comparison ===")
    for method in ("tokenize", "sum"):
        v_ed = get_embedding(tokenizer, embeddings, "ed",   method=method)
        v_jump = get_embedding(tokenizer, embeddings, "jump", method=method)
        query = v_jump+v_ed

        rank = get_word_rank(tokenizer, embeddings, query, "jumped", method=method)
        print(f"\n method={method}: rank of 'jumped' = {int(rank)}")
        print("  top-5:", [(tok, f"{score:.4f}") for tok, score in find_closest(tokenizer, embeddings, query, top_k=5)])

if __name__ == "__main__":
    main()