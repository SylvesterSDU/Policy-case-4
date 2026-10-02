"""
Builds and caches embeddings for every policy in the database, so
Method 3 never has to re-embed the same 98 policies more than once.
Run this once (or whenever company_policies.csv changes); method3
loads the cached file instead of re-calling the embeddings API.
"""

import json

from llm_common import get_client, load_policies

EMBEDDING_MODEL = "gemini-embedding-001"
CACHE_PATH = "policy_embeddings_cache.json"


def build_index():
    client = get_client()
    policies = load_policies()

    # Embed title + text together, in reasonably sized batches, so we
    # need only a handful of API calls for all 98 policies rather than 98.
    texts = [f"{p['title']}: {p['policy_text']}" for p in policies]

    batch_size = 20
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.models.embed_content(model=EMBEDDING_MODEL, contents=batch)
        all_embeddings.extend([e.values for e in response.embeddings])
        print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} policies...")

    cache = []
    for p, emb in zip(policies, all_embeddings):
        cache.append({
            "title": p["title"],
            "department": p["department"],
            "category": p["category"],
            "policy_text": p["policy_text"],
            "embedding": emb,
        })

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    print(f"Saved {len(cache)} policy embeddings -> {CACHE_PATH}")


if __name__ == "__main__":
    build_index()
