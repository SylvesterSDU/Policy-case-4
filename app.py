import csv
import os

from flask import Flask, render_template

app = Flask(__name__)

CSV_PATH = os.path.join(os.path.dirname(__file__), "comparison_full.csv")

# The single correct policy each question was designed around (used only
# to compute an accuracy summary for the page - q8 has no single clean
# right answer, since three real policies are all plausibly relevant, so
# it's excluded from the accuracy count but still shown in the raw table).
EXPECTED_POLICY = {
    "q1": "Dress Code Policy", "q2": "Warranty Policy", "q3": "Performance Review Policy",
    "q4": "Password Policy", "q5": "Vacation Policy", "q6": "Work From Abroad Policy",
    "q7": "Security Incident Policy", "q9": "Whistleblower Policy",
    "q10": "Expense Approval Policy",
}

METHOD_LABELS = {
    "rules_based": "Rules-based search",
    "llm_no_index": "LLM, no vector index",
    "llm_with_index": "LLM + vector index",
}


def load_rows():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def build_summary(rows):
    summary = {}
    for key in METHOD_LABELS:
        times, tokens, correct, scored, unsupported = [], [], 0, 0, 0
        for r in rows:
            t = r.get(f"{key}_time_ms")
            if t:
                times.append(float(t))
            tok = r.get(f"{key}_tokens")
            if tok:
                tokens.append(float(tok))

            cited = r.get(f"{key}_policy", "")
            qid = r["id"]
            if qid in EXPECTED_POLICY:
                scored += 1
                if cited == EXPECTED_POLICY[qid]:
                    correct += 1
            elif r["category"] == "unanswerable":
                scored += 1
                if cited == "":
                    correct += 1
                else:
                    unsupported += 1

        summary[key] = {
            "label": METHOD_LABELS[key],
            "avg_time_ms": round(sum(times) / len(times), 0) if times else 0,
            "avg_tokens": round(sum(tokens) / len(tokens), 0) if tokens else None,
            "accuracy": f"{correct}/{scored}",
            "unsupported_on_trick_qs": unsupported,
        }
    return summary


@app.route("/")
def index():
    rows = load_rows()
    summary = build_summary(rows)
    return render_template("index.html", rows=rows, summary=summary, methods=METHOD_LABELS)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
