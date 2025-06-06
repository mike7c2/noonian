from ollama import chat
from ollama import ChatResponse
from . import QueueShutdownException

from .logger_config import setup_logger
logger = setup_logger(__name__)

def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
        a (int): The first number
        b (int): The second number

    Returns:
        int: The sum of the two numbers
    """

    # The cast is necessary as returned tool call arguments don't always conform exactly to schema
    # E.g. this would prevent "what is 30 + 12" to produce '3012' instead of 42
    return int(a) + int(b)


def subtract_two_numbers(a: int, b: int) -> int:
    """
    Subtract two numbers
    """

    # The cast is necessary as returned tool call arguments don't always conform exactly to schema
    return int(a) - int(b)

available_functions = {
    'add_two_numbers': add_two_numbers,
    'subtract_two_numbers': subtract_two_numbers,
}

def do_tool_calls(tool_calls):
    logger.info(f"Processing tool calls: {tool_calls}")
    messages = []
    for tool in tool_calls:
        if function_to_call := available_functions.get(tool.function.name):
            output = function_to_call(**tool.function.arguments)
            messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})
        else:
            logger.info('Function', tool.function.name, 'not found')
    return messages

def handle_llm_queries(args, stop_event, text_snippet_queue, llm_response_queue):
    system_message = {"role": "system", "content": args.ollama_system_prompt}
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
            logger.info(f"Got prompt: {new_text}")

            while True:
                response: ChatResponse = chat(
                    model=args.ollama_model,
                    messages=messages,
                    tools=available_functions.values(),
                    stream=True
                )
                output_buf = ""
                tool_called = False
                for chunk in response:
                    print(chunk)
                    if chunk.message.tool_calls is not None:
                        messages += do_tool_calls(chunk.message.tool_calls)
                        tool_called = True
                    if len(chunk.message.content) > 0:
                        output_buf += chunk.message.content
                        llm_response_queue.put(chunk.message.content)

                if len(output_buf) > 0:
                    new_message = {
                        'role': 'assistant',
                        'content': output_buf,
                    }
                    messages.append(new_message)
                    logger.info(new_message)

                if tool_called:
                    continue
                else:
                    break
                    

    except QueueShutdownException:
        pass
