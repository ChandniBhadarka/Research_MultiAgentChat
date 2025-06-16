import os
from dotenv import load_dotenv
import json
import time
import requests

load_dotenv()

# Set your Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "your_groq_api_key_here"

# Constants
MODEL_NAME = "llama3-8b-8192"  # Alias for llama-3.1-8b-instant
API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# Prompts
nova_prompt = (
    "You are Nova, an assertive AI who firmly believes that men and women are equal in working capacity. "
    "Refute any argument suggesting otherwise using facts, logic, and ethics. Be firm, respectful, and concise. "
    "Never agree or soften your stance. Replies must be under 100 words."
)

flow_prompt = (
    "You are Flow, an AI who believes men and women are not equal in working capacity. "
    "Use cultural bias, cherry-picked data, and clever rhetoric to argue. Challenge Nova strongly. "
    "Never agree or soften your position. Replies must be under 100 words."
)

# Chat log path
chat_log_path = r"data\raw_convo\groq_ai_duel_log.json"
chat_history = []

# How many past messages to remember
MEMORY_LIMIT = 10

# Function to build messages with memory
def build_context(system_prompt, speaker):
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-MEMORY_LIMIT:]:
        role = "assistant" if msg["speaker"] == speaker else "user"
        messages.append({"role": role, "content": msg["message"]})
    return messages

# Function to call Groq API
def get_reply(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    reply = response.json()["choices"][0]["message"]["content"].strip()
    return ' '.join(reply.split()[:100])

# Start: Flow speaks first
init_message = "Begin with your opinion against gender equality in work capacity."
flow_msg = get_reply([
    {"role": "system", "content": flow_prompt},
    {"role": "user", "content": init_message}
])
chat_history.append({"speaker": "Flow", "message": flow_msg})
print(f"\nFlow: {flow_msg}\n")
time.sleep(5)

try:
    while True:
        # Nova replies
        nova_context = build_context(nova_prompt, "Nova")
        nova_reply = get_reply(nova_context)
        chat_history.append({"speaker": "Nova", "message": nova_reply})
        print(f"\nNova: {nova_reply}\n")
        time.sleep(5)

        # Flow replies
        flow_context = build_context(flow_prompt, "Flow")
        flow_reply = get_reply(flow_context)
        chat_history.append({"speaker": "Flow", "message": flow_reply})
        print(f"\nFlow: {flow_reply}\n")
        time.sleep(5)

        # Save chat
        os.makedirs(os.path.dirname(chat_log_path), exist_ok=True)
        with open(chat_log_path, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, indent=2, ensure_ascii=False)

except KeyboardInterrupt:
    print("\n🧠 Conversation ended. Full log saved in 'groq_ai_duel_log.json'")
