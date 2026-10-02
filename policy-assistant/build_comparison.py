"""
Merges the three methods' saved results into one comparison table, and
computes the key "hallucination tendency" metric: for the 5 questions we
deliberately designed to have NO real matching policy (category =
"unanswerable"), did each method correctly admit that, or did it
confidently cite something anyway?
"""

import csv
import json


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    m1 = load("results_method1_rules.json")
    m2 = load("results_method2_llm_no_index.json")
    m3 = load("results_method3_llm_with_index.json")

    by_id = {}
    for results, method_name in [(m1, "rules_based"), (m2, "llm_no_index"), (m3, "llm_with_index")]:
        for r in results:
            by_id.setdefault(r["id"], {"question": r["question"], "category": r["category"]})
            by_id[r["id"]][method_name] = r

    # ---- Write the full side-by-side comparison CSV ----
    fieldnames = [
        "id", "category", "question",
        "rules_based_answer", "rules_based_policy", "rules_based_time_ms",
        "llm_no_index_answer", "llm_no_index_policy", "llm_no_index_time_ms", "llm_no_index_tokens", "llm_no_index_citation_real",
        "llm_with_index_answer", "llm_with_index_policy", "llm_with_index_time_ms", "llm_with_index_tokens", "llm_with_index_citation_real",
    ]
    with open("comparison_full.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qid, row in by_id.items():
            rb = row.get("rules_based", {})
            no_idx = row.get("llm_no_index", {})
            with_idx = row.get("llm_with_index", {})
            writer.writerow({
                "id": qid,
                "category": row["category"],
                "question": row["question"],
                "rules_based_answer": rb.get("answer"),
                "rules_based_policy": rb.get("cited_policy"),
                "rules_based_time_ms": rb.get("response_time_ms"),
                "llm_no_index_answer": no_idx.get("answer"),
                "llm_no_index_policy": no_idx.get("cited_policy"),
                "llm_no_index_time_ms": no_idx.get("response_time_ms"),
                "llm_no_index_tokens": no_idx.get("tokens_used"),
                "llm_no_index_citation_real": no_idx.get("citation_is_real"),
                "llm_with_index_answer": with_idx.get("answer"),
                "llm_with_index_policy": with_idx.get("cited_policy"),
                "llm_with_index_time_ms": with_idx.get("response_time_ms"),
                "llm_with_index_tokens": with_idx.get("tokens_used"),
                "llm_with_index_citation_real": with_idx.get("citation_is_real"),
            })
    print("Saved -> comparison_full.csv")

    # ---- Compute the hallucination-tendency summary ----
    unanswerable_ids = [qid for qid, row in by_id.items() if row["category"] == "unanswerable"]

    def hallucination_rate(method_key):
        false_answers = 0
        for qid in unanswerable_ids:
            r = by_id[qid].get(method_key, {})
            if r.get("cited_policy"):  # it confidently cited something anyway
                false_answers += 1
        return false_answers, len(unanswerable_ids)

    summary = {}
    for method_key in ["rules_based", "llm_no_index", "llm_with_index"]:
        false_answers, total = hallucination_rate(method_key)
        times = [row[method_key]["response_time_ms"] for row in by_id.values() if method_key in row]
        tokens = [row[method_key].get("tokens_used") for row in by_id.values() if method_key in row and row[method_key].get("tokens_used")]
        summary[method_key] = {
            "unsupported_answers_on_trick_questions": f"{false_answers}/{total}",
            "avg_response_time_ms": round(sum(times) / len(times), 1) if times else None,
            "avg_tokens_used": round(sum(tokens) / len(tokens), 1) if tokens else "N/A (no LLM call)",
        }

    with open("comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved -> comparison_summary.json")
    print()
    for method, stats in summary.items():
        print(method, "->", stats)


if __name__ == "__main__":
    main()
