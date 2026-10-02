"""Shared helpers for the two LLM-based methods (with and without a
vector index). Keeping this in one place means both methods use the
exact same prompt structure, JSON schema, and generation model - so the
only variable that differs between them is HOW MANY policies get fed
into the prompt (all 98 vs. only the top few retrieved by similarity).
"""

import csv
import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
GENERATION_MODEL = "gemini-3.1-flash-lite"


def get_client():
    if not API_KEY:
        raise SystemExit("No GEMINI_API_KEY found. Create a .env file with GEMINI_API_KEY=your_key")
    return genai.Client(api_key=API_KEY)


def load_policies(path="company_policies.csv"):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def format_policies_block(policies):
    lines = []
    for p in policies:
        lines.append(f"- Title: \"{p['title']}\" | Department: {p['department']} | Category: {p['category']} | Text: {p['policy_text']}")
    return "\n".join(lines)


PROMPT_TEMPLATE = """You are a company policy assistant. You are given a list of the ONLY policies that exist in the company's policy database. Answer the employee's question using ONLY the information in these policies.

POLICIES:
{policies_block}

RULES:
- If one of the policies above answers the question, answer it and cite that policy's EXACT title (copy it exactly as written above, including capitalization).
- If none of the policies above are actually relevant to the question, you MUST say so - set "cited_policy" to null and "answer" to a short honest statement that no policy covers this. Do NOT guess, speculate, or make up a policy that isn't listed above.
- Do not invent any policy details that are not explicitly stated in the text above.

Respond with ONLY a JSON object in this exact format, no markdown fences, no extra text:
{{"answer": "your answer here", "cited_policy": "Exact Policy Title" or null}}

EMPLOYEE QUESTION: "{question}"
"""


def ask_llm(client, question, policies_subset, max_retries=3):
    policies_block = format_policies_block(policies_subset)
    prompt = PROMPT_TEMPLATE.format(policies_block=policies_block, question=question)

    start = time.time()
    response = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            break
        except Exception as e:
            is_transient = "UNAVAILABLE" in str(e) or "high demand" in str(e).lower()
            if is_transient and attempt < max_retries - 1:
                print("  (model temporarily unavailable, retrying in 10s...)", end=" ")
                time.sleep(10)
            else:
                raise
    elapsed_ms = round((time.time() - start) * 1000, 2)

    try:
        parsed = json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        parsed = {"answer": response.text, "cited_policy": None}

    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    output_tokens = getattr(usage, "candidates_token_count", None) if usage else None

    return {
        "answer": parsed.get("answer"),
        "cited_policy": parsed.get("cited_policy"),
        "response_time_ms": elapsed_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_used": (input_tokens or 0) + (output_tokens or 0) if usage else None,
    }


def citation_is_real(cited_policy, policies):
    if not cited_policy:
        return None  # no citation was made, N/A rather than a fabricated one
    real_titles = {p["title"] for p in policies}
    return cited_policy in real_titles
