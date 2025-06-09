import ollama
import json
import time

# ========== CONFIGURATION ==========
MODEL_NAME = 'gemma2:2b'
MAX_HISTORY_TURNS = 30  # keep recent turns to fit context window
LOG_FILE = "conversation_modified_prompt.jsonl"
MAX_WORDS = 100          # Limit output length per your requirement
DELAY_BETWEEN_TURNS = 1  # seconds delay for readability

# ========== LOAD PROMPTS ==========
def load_prompt(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read().strip()

AGENT_1_NAME = "Winslop"
AGENT_2_NAME = "Finista"

AGENT_1_ROLE = load_prompt("winslop_modified_prompt.txt")
AGENT_2_ROLE = load_prompt("finista_modified_prompt.txt")

# ========== CHAT HISTORY ==========
history = []

# ========== UTILITY FUNCTIONS ==========

def truncate_history(history, max_turns):
    # Keep only the last max_turns messages to avoid exceeding token limit
    return history[-max_turns:]

def build_prompt(agent_role, history):
    # Build prompt with role + chat history
    prompt = f"{agent_role}\n\nConversation so far:\n"
    for msg in history:
        prompt += f"{msg['speaker']}: {msg['text']}\n"
    prompt += "\nYour response (max 100 words):"
    return prompt.strip()

def limit_to_100_words(text):
    words = text.split()
    if len(words) > MAX_WORDS:
        return " ".join(words[:MAX_WORDS])
    return text

def chat_with_agent(prompt):
    response = ollama.chat(model=MODEL_NAME, messages=[
        {'role': 'user', 'content': prompt}
    ])
    # Extract content and limit to 100 words
    content = response['message']['content'].strip()
    return limit_to_100_words(content)

def take_turn(agent_name, agent_role):
    global history
    history = truncate_history(history, MAX_HISTORY_TURNS)
    prompt = build_prompt(agent_role, history)
    response = chat_with_agent(prompt)
    history.append({'speaker': agent_name, 'text': response})
    return response

def log_turn(agent_name, response):
    log_entry = {
        'agent': agent_name,
        'response': response,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + "\n")

# ========== START LOOP ==========

print("==== Infinite AI Chat Between Two Agents (Ctrl+C to stop) ====\n")

turn_counter = 1
try:
    while True:
        print(f"\n========= Turn {turn_counter} =========")

        if turn_counter % 2 == 1:
            agent_name = AGENT_1_NAME
            agent_role = AGENT_1_ROLE
        else:
            agent_name = AGENT_2_NAME
            agent_role = AGENT_2_ROLE

        response = take_turn(agent_name, agent_role)
        print(f"{agent_name}: {response}\n")
        log_turn(agent_name, response)

        turn_counter += 1
        time.sleep(DELAY_BETWEEN_TURNS)

except KeyboardInterrupt:
    print("\n\n==== Chat Ended by User (Ctrl+C) ====")
