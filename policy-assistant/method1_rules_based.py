"""
Method 1: Rules-based search (no LLM).

Scores every policy by how many meaningful words it shares with the
question (simple token-overlap scoring), and returns the best match.
If nothing overlaps at all, it honestly says no policy was found rather
than guessing - this method structurally cannot "hallucinate" content,
since it only ever quotes real policy text back, but it CAN fail to find
a genuinely relevant policy that uses different wording.
"""

import csv
import json
import re
import time

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "when", "where", "who", "how", "why", "which", "do", "does",
    "did", "can", "could", "would", "should", "will", "shall", "may",
    "might", "must", "i", "we", "you", "he", "she", "it", "they", "my",
    "our", "your", "his", "her", "its", "their", "to", "of", "in", "on",
    "at", "for", "with", "about", "if", "there", "s", "get", "have",
    "has", "had", "and", "or", "but", "not", "this", "that", "these",
    "those", "am", "allowed", "policy",
}


def tokenize(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def load_policies(path="company_policies.csv"):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_policy_tokens(policies):
    """Pre-tokenize each policy once (title + text + department + category)."""
    for p in policies:
        combined = f"{p['title']} {p['policy_text']} {p['department']} {p['category']}"
        p["_tokens"] = tokenize(combined)
    return policies


def search(question, policies):
    q_tokens = tokenize(question)
    scored = []
    for p in policies:
        overlap = q_tokens & p["_tokens"]
        scored.append((len(overlap), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


MIN_MATCH_SCORE = 2  # below this, treat as "no relevant policy found" rather
                      # than citing a weakly, coincidentally overlapping one


def answer_question(question, policies):
    start = time.time()
    scored = search(question, policies)
    elapsed_ms = round((time.time() - start) * 1000, 2)

    best_score, best_policy = scored[0]

    if best_score < MIN_MATCH_SCORE:
        return {
            "answer": "No matching policy was found in the database for this question.",
            "cited_policy": None,
            "response_time_ms": elapsed_ms,
            "tokens_used": None,  # no LLM call, not applicable
            "match_score": best_score,
        }

    return {
        "answer": f"According to the {best_policy['title']}: {best_policy['policy_text']}",
        "cited_policy": best_policy["title"],
        "response_time_ms": elapsed_ms,
        "tokens_used": None,
        "match_score": best_score,
    }


def main():
    policies = build_policy_tokens(load_policies())
    questions = json.load(open("test_questions.json"))

    results = []
    for q in questions:
        result = answer_question(q["question"], policies)
        result["id"] = q["id"]
        result["category"] = q["category"]
        result["question"] = q["question"]
        result["method"] = "rules_based"
        results.append(result)
        print(f"[{q['id']}] {q['question']}")
        print(f"  -> {result['cited_policy'] or 'NO MATCH'} (score={result['match_score']})")

    with open("results_method1_rules.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> results_method1_rules.json")


if __name__ == "__main__":
    main()
