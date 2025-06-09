import ollama
import json
import time

# ========== CONFIGURATION ==========
MODEL_NAME = 'gemma2:2b'
MAX_HISTORY_TURNS = 30  # You can adjust based on token budget
LOG_FILE = "conversation_prompt9.jsonl"

# ========== LOAD PROMPTS ==========
def load_prompt(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read().strip()

AGENT_1_NAME = "Winslop"
AGENT_2_NAME = "Finista"

AGENT_1_ROLE = load_prompt("agent1_prompt9.txt")
AGENT_2_ROLE = load_prompt("agent2_prompt9.txt")

# ========== CHAT HISTORY ==========
history = []

# ========== UTILITY FUNCTIONS ==========

def truncate_history(history, max_turns):
    return history[-max_turns:]

def build_prompt(agent_role, history):
    prompt = f"{agent_role}\n\nConversation so far:\n"
    for msg in history:
        prompt += f"{msg['speaker']}: {msg['text']}\n"
    prompt += "\nYour response:"
    return prompt.strip()

def chat_with_agent(prompt):
    response = ollama.chat(model=MODEL_NAME, messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content'].strip()

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

except KeyboardInterrupt:
    print("\n\n==== Chat Ended by User (Ctrl+C) ====")
