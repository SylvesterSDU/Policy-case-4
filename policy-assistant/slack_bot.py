"""
Slack bot for the Company Policy Assistant.

Reuses the exact same retrieval + LLM logic as method3_llm_with_index.py
(the best-performing method from the comparison: embeds the question,
retrieves the top-5 most similar policies, and only sends those to the
LLM). This file just adds a Slack layer on top - no new answering logic.

Run locally with Socket Mode, so no public URL / hosting is needed.

Setup:
1. Place this file in the SAME folder as method3_llm_with_index.py,
   llm_common.py, policy_embeddings_cache.json, and company_policies.csv
   (i.e. your `policy-assistant/` folder).
2. Add two new variables to your existing .env file:
     SLACK_BOT_TOKEN=xoxb-...
     SLACK_APP_TOKEN=xapp-...
   (keep your existing GEMINI_API_KEY line too)
3. pip install slack_bolt --break-system-packages   (or plain pip install slack_bolt)
4. python3 slack_bot.py
5. In Slack, invite the bot to your channel: /invite @Policy Assistant Bot
6. Ask it a question by @mentioning it, e.g.:
     @Policy Assistant Bot how long is the product warranty?
"""

import os
import re

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from llm_common import get_client
from method3_llm_with_index import load_index, answer_question

load_dotenv()

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")

if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    raise SystemExit(
        "Missing SLACK_BOT_TOKEN or SLACK_APP_TOKEN. Add both to your .env file."
    )

# Set up the same client + policy index the comparison script used.
gemini_client = get_client()
policy_cache, policy_embeddings = load_index()

app = App(token=SLACK_BOT_TOKEN)


def strip_mention(text):
    """Remove the leading '<@BOTID>' Slack inserts before the question text."""
    return re.sub(r"^<@[^>]+>\s*", "", text).strip()


@app.event("app_mention")
def handle_mention(event, say):
    question = strip_mention(event.get("text", ""))

    if not question:
        say("Ask me a policy question, e.g. `@Policy Assistant Bot how long is the product warranty?`")
        return

    say(text="Looking that up...", thread_ts=event["ts"])

    try:
        result = answer_question(gemini_client, question, policy_cache, policy_embeddings)
    except Exception as e:
        say(text=f"Sorry, something went wrong answering that: `{e}`", thread_ts=event["ts"])
        return

    answer = result.get("answer") or "I couldn't find an answer."
    cited_policy = result.get("cited_policy")

    if cited_policy:
        reply = f"{answer}\n\n:page_facing_up: *Relevant policy:* {cited_policy}"
    else:
        reply = f"{answer}\n\n_No specific policy in the database directly covers this._"

    say(text=reply, thread_ts=event["ts"])


if __name__ == "__main__":
    print("Policy Assistant Bot is starting (Socket Mode)...")
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
