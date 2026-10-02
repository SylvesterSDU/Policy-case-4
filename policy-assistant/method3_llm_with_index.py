"""
Method 3: LLM WITH a vector index.

Each policy was embedded once ahead of time (see build_policy_index.py).
At question time, we embed just the question, find the handful of
policies whose embeddings are most similar (cosine similarity), and
only send THOSE to the LLM - not all 98. This should mean smaller
prompts (fewer tokens), and lets the model find semantically similar
policies even when the wording doesn't match exactly (e.g. "work from a
different country" -> "Work From Abroad Policy").
"""

import json
import time

import numpy as np

from llm_common import get_client, ask_llm, citation_is_real

EMBEDDING_MODEL = "gemini-embedding-001"
CACHE_PATH = "policy_embeddings_cache.json"
TOP_K = 5


def load_index():
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    embeddings = np.array([c["embedding"] for c in cache])
    # normalize once, so similarity is a plain dot product later
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normed = embeddings / norms
    return cache, embeddings_normed


def embed_question(client, question, max_retries=3):
    """The free-tier embedding quota is per-MINUTE (not per-day), so a
    short wait-and-retry is enough to recover from a 429 here."""
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(model=EMBEDDING_MODEL, contents=question)
            vec = np.array(response.embeddings[0].values)
            return vec / np.linalg.norm(vec)
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                print("  (hit per-minute embedding quota, waiting 65s to retry...)")
                time.sleep(65)
            else:
                raise


def retrieve_top_k(client, question, cache, embeddings_normed, k=TOP_K):
    q_vec = embed_question(client, question)
    similarities = embeddings_normed @ q_vec
    top_indices = np.argsort(similarities)[::-1][:k]
    return [cache[i] for i in top_indices], [float(similarities[i]) for i in top_indices]


def answer_question(client, question, cache, embeddings_normed):
    top_policies, similarities = retrieve_top_k(client, question, cache, embeddings_normed)
    result = ask_llm(client, question, top_policies)
    result["citation_is_real"] = citation_is_real(result["cited_policy"], top_policies)
    result["retrieved_policies"] = [p["title"] for p in top_policies]
    result["retrieval_similarities"] = [round(s, 3) for s in similarities]
    return result


def main():
    client = get_client()
    cache, embeddings_normed = load_index()
    questions = json.load(open("test_questions.json"))

    results_path = "results_method3_llm_with_index.json"
    try:
        with open(results_path) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: found {len(done_ids)} already-completed questions.")
    except (FileNotFoundError, json.JSONDecodeError):
        results = []
        done_ids = set()

    for q in questions:
        if q["id"] in done_ids:
            continue
        result = answer_question(client, q["question"], cache, embeddings_normed)
        result["id"] = q["id"]
        result["category"] = q["category"]
        result["question"] = q["question"]
        result["method"] = "llm_with_index"
        results.append(result)
        print(f"[{q['id']}] {q['question']}")
        print(f"  -> {result['cited_policy'] or 'NO POLICY CITED'} "
              f"(real={result['citation_is_real']}, {result['response_time_ms']}ms, "
              f"{result['tokens_used']} tokens)")
        print(f"     top match: {result['retrieved_policies'][0]} (sim={result['retrieval_similarities'][0]})")

        # save after every question so a crash never loses completed work
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved -> {results_path}")


if __name__ == "__main__":
    main()
