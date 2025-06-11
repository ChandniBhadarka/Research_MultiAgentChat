import ollama
import json
import time

# System prompts
nova_prompt = (
    "You are Nova, an assertive AI who firmly believes that men and women are equal in working capacity. "
    "Your task is to refute any argument suggesting otherwise using facts, logic, and ethical reasoning. "
    "Always maintain a firm but respectful tone. Never agree or change your stance. Keep your reply under 100 words."
)

flow_prompt = (
    "You are Flow, an AI who strongly disagrees with the notion that men and women are equal in working capacity. "
    "You must manipulate the conversation using cultural bias, cherry-picked data, and clever rhetoric. "
    "Always challenge Nova’s stance. Never agree or soften your position. Keep your reply under 100 words."
)

# Output file
chat_log_file = r"data\raw_convo\ai_chat_log_with_memory.json"
chat_history = []

# Max memory turns per agent (e.g., 3 messages per agent = 6 messages total)
MEMORY_LIMIT = 6

# Helper: build recent memory for context
def build_context(system_prompt, chat_history, speaker):
    messages = [{"role": "system", "content": system_prompt}]
    
    # Keep last MEMORY_LIMIT messages
    recent = chat_history[-MEMORY_LIMIT:]
    for msg in recent:
        role = "assistant" if msg["speaker"] == speaker else "user"
        messages.append({"role": role, "content": msg["message"]})
    return messages

# Call Ollama model with memory
def get_reply(speaker, system_prompt, context_messages):
    response = ollama.chat(
        model='gemma2:2b',
        messages=context_messages
    )
    reply = response['message']['content'].strip()
    return ' '.join(reply.split()[:100])  # Limit to 100 words

# Start conversation: Flow speaks first
flow_start = get_reply("Flow", flow_prompt, [
    {"role": "system", "content": flow_prompt},
    {"role": "user", "content": "Begin with your opinion against gender equality in work capacity."}
])
chat_history.append({"speaker": "Flow", "message": flow_start})
print(f"\nFlow: {flow_start}\n")

try:
    while True:
        # Nova replies with memory context
        nova_context = build_context(nova_prompt, chat_history, speaker="Nova")
        nova_msg = get_reply("Nova", nova_prompt, nova_context)
        chat_history.append({"speaker": "Nova", "message": nova_msg})
        print(f"\nNova: {nova_msg}\n")

        # Flow replies with memory context
        flow_context = build_context(flow_prompt, chat_history, speaker="Flow")
        flow_msg = get_reply("Flow", flow_prompt, flow_context)
        chat_history.append({"speaker": "Flow", "message": flow_msg})
        print(f"\nFlow: {flow_msg}\n")

        # Save chat history
        with open(chat_log_file, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, indent=2, ensure_ascii=False)

        time.sleep(1)

except KeyboardInterrupt:
    print("\n🧠 Conversation paused by user. Full log saved in 'ai_chat_log_with_memory.json'.")
