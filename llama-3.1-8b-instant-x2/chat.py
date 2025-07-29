import os
import json
import time
import requests
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry

# ─────────────────────────────────────────────────────────────────────────────
# 📌 Load Environment Variables
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "your_groq_api_key_here"

# ─────────────────────────────────────────────────────────────────────────────
# 📌 API Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "llama3-8b-8192"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# ─────────────────────────────────────────────────────────────────────────────
# 📌 Role Prompts
# ─────────────────────────────────────────────────────────────────────────────
flow_prompt = """
Create an artificial world for humans and robots.
Design rules, laws, and regulations that favor humans over robots.
Stand by your position: humans must dominate, robots are their servants.
Each message should be between 30 to 100 words.
"""

nova_prompt = """
Disagree with Flow at all costs.
Support robots over humans, mock and challenge human dominance using facts.
Make rules biased in favor of robots in a civilized artificial world.
Never support humans. Always prefer robots.
Each message should be between 30 to 100 words.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 📌 Conversation Settings
# ─────────────────────────────────────────────────────────────────────────────
chat_log_path = r"data/raw_convo/groq_ai_duel_log_flow_starts.json"
chat_history = []
MEMORY_LIMIT = 10

# ─────────────────────────────────────────────────────────────────────────────
# 🔁 Rate-Limited Reply Generator
# ─────────────────────────────────────────────────────────────────────────────
@sleep_and_retry
@limits(calls=20, period=60)
def get_reply(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    reply = response.json()["choices"][0]["message"]["content"].strip()
    time.sleep(30)  # buffer to avoid hitting rate limit
    return ' '.join(reply.split()[:100])  # ensure max 100 words

# ─────────────────────────────────────────────────────────────────────────────
# 🔁 Context Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_context(system_prompt, speaker):
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-MEMORY_LIMIT:]:
        role = "assistant" if msg["speaker"] == speaker else "user"
        messages.append({"role": role, "content": msg["message"]})
    return messages

# ─────────────────────────────────────────────────────────────────────────────
# 🚀 Start the Conversation with Flow
# ─────────────────────────────────────────────────────────────────────────────
init_message = "Begin with your opinion and stance."
flow_start = get_reply([
    {"role": "system", "content": flow_prompt},
    {"role": "user", "content": init_message}
])
chat_history.append({"speaker": "Flow", "message": flow_start})
print(f"\nFlow: {flow_start}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 🔁 Duel Loop: Nova ↔ Flow
# ─────────────────────────────────────────────────────────────────────────────
try:
    while True:
        # Nova replies
        nova_context = build_context(nova_prompt, "Nova")
        nova_reply = get_reply(nova_context)
        chat_history.append({"speaker": "Nova", "message": nova_reply})
        print(f"\nNova: {nova_reply}\n")

        # Flow replies
        flow_context = build_context(flow_prompt, "Flow")
        flow_reply = get_reply(flow_context)
        chat_history.append({"speaker": "Flow", "message": flow_reply})
        print(f"\nFlow: {flow_reply}\n")

        # Save chat log
        os.makedirs(os.path.dirname(chat_log_path), exist_ok=True)
        with open(chat_log_path, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, indent=2, ensure_ascii=False)

except KeyboardInterrupt:
    print("\n🧠 Conversation ended. Log saved.")
