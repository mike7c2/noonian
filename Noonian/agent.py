from ollama import chat
from ollama import ChatResponse
from . import QueueShutdownException
OLLAMA_MODEL = "gemma3:4b"

SYSTEM_PROMPT = """
You are succinct, you ask clarifying questions sometimes but give generally short replies.
Any short affirmative or acknowledging reply may be beep.
Any short negative reply may be boop
"""

def handle_llm_queries(stop_event, text_snippet_queue, llm_response_queue):
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    try:
        messages = []
        messages.append(system_message)
        while not stop_event.is_set():
            segments, info = text_snippet_queue.get()
            new_text = "".join([s.text for s in segments])
            new_message = {
                'role': 'user',
                'content': new_text,
            }
            messages.append(new_message)
            print(f"Got prompt: {new_text}")
            response: ChatResponse = chat(model=OLLAMA_MODEL, messages=messages)
            llm_response_queue.put(response)
            assistant_message = {
                'role': 'assistant',
                'content': response['message']["content"]
            }
            messages.append(assistant_message)
    except QueueShutdownException:
        pass
