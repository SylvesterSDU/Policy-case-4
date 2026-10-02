"""
Method 2: LLM WITHOUT a vector index.

Every question gets the FULL list of all 98 policies stuffed into the
prompt, and the LLM has to find the relevant one itself, using only its
own reading of the full list (no retrieval step, no pre-filtering).
This is simple to build, but scales badly - the entire policy database
has to be re-sent (and re-read by the model) on every single question.
"""

import json

from llm_common import get_client, load_policies, ask_llm, citation_is_real

MODEL_NAME = "gemini-3.1-flash-lite"


def answer_question(client, question, all_policies):
    result = ask_llm(client, question, all_policies)
    result["citation_is_real"] = citation_is_real(result["cited_policy"], all_policies)
    return result


def main():
    client = get_client()
    policies = load_policies()
    questions = json.load(open("test_questions.json"))

    results = []
    for q in questions:
        result = answer_question(client, q["question"], policies)
        result["id"] = q["id"]
        result["category"] = q["category"]
        result["question"] = q["question"]
        result["method"] = "llm_no_index"
        results.append(result)
        print(f"[{q['id']}] {q['question']}")
        print(f"  -> {result['cited_policy'] or 'NO POLICY CITED'} "
              f"(real={result['citation_is_real']}, {result['response_time_ms']}ms, "
              f"{result['tokens_used']} tokens)")

    with open("results_method2_llm_no_index.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> results_method2_llm_no_index.json")


if __name__ == "__main__":
    main()
