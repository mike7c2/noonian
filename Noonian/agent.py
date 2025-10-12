import json
import queue
import threading
from typing import List, Any, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

import signal
import asyncio
import argparse
import os
import logging
import paho.mqtt.client as mqtt

# Set up logger
from .logger_config import setup_logger
logger = setup_logger(__name__)

# Define the state schema for the agent
class AgentState(TypedDict):
    messages: List[Any]
    next: str

# Define the agent executor that processes the state
def agent_node(state, agent):
    messages = state["messages"]
    logger.debug(f"Agent processing messages: {messages}")
    result = agent.invoke(messages)
    logger.debug(f"Agent output: {result}")
    return {"messages": messages + [result]}

# Define output processing node
def output_node(state, response_queue):
    messages = state["messages"]
    return {"messages": messages, "next": END}

# Router function to determine the next node
def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    # Check for tool_calls as an attribute (LangChain AIMessage)
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tool"
    else:
        return "END"

def build_mcp_config(tool_files=None, tool_servers=None) -> Dict:
    """Build MCP client configuration from tool files and server URLs."""
    config = {}
    
    # Add file-based tools
    if tool_files:
        for i, file_path in enumerate(tool_files):
            if not os.path.exists(file_path):
                logger.warning(f"Tool file not found: {file_path}")
                continue
                
            tool_name = f"file_tool_{i}"
            config[tool_name] = {
                "command": "python",
                "args": [os.path.abspath(file_path)],
                "transport": "stdio",
            }
            logger.info(f"Added file-based tool: {tool_name} from {file_path}")
            
    # Add server-based tools
    if tool_servers:
        for i, server_url in enumerate(tool_servers):
            tool_name = f"server_tool_{i}"
            config[tool_name] = {
                "url": server_url,
                "transport": "streamable_http",
            }
            logger.info(f"Added server-based tool: {tool_name} from {server_url}")
            
    return config

class NoonianAgent:
    def __init__(self, args):
        self.args = args
        self.context = []
        # Ensure the async graph is initialized synchronously for sync usage
        loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else asyncio.new_event_loop()
        if not loop.is_running():
            asyncio.set_event_loop(loop)
        loop.run_until_complete(self._init_graph())

    async def _init_graph(self):
        # Build MCP client configuration from args
        mcp_config = build_mcp_config(
            tool_files=self.args.tool_file,
            tool_servers=self.args.tool_server
        )
        
        client = MultiServerMCPClient(mcp_config)

        # Initialise the LLM and bind tools
        llm = ChatOllama(model=self.args.ollama_model)
        tools = await client.get_tools()

        if not tools:
            logger.warning("No tools were loaded. The agent will function as a basic chat assistant.")
        else:
            logger.info(f"Loaded {len(tools)} tools")
            
        self.agent = llm.bind_tools(tools)

        # Create the graph
        workflow = StateGraph(AgentState)

        # Create nodes
        workflow.add_node("agent", lambda state: agent_node(state, self.agent))
        async def tool_node_async(state):
            tool_result = await ToolNode(tools).ainvoke(state)
            return {**state, "messages": state["messages"] + tool_result["messages"]}
        workflow.add_node("tool", tool_node_async)
        # Create routing
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            router,
            {
                "tool": "tool",
                "END": END
            }
        )
        workflow.add_edge("tool", "agent")

        self.app = workflow.compile()

    async def handle_llm_query(self, query):
        messages = [SystemMessage(content=self.args.ollama_system_prompt)] + self.context
        messages.append(HumanMessage(content=query))
        logging.debug(f"Input: {messages}")
        result = await self.app.ainvoke({"messages": messages, "next": "agent"})
        logging.debug(f"Output: {result}")
        self.context += result["messages"]
        return result

    def clear_context(self):
        self.context = []

# Runs a noonian agent taking from text_snippet_queue and putting to llm_response_queue
def noonian_agent_runner(args, stop_flag, text_snippet_queue, mqtt_client):
    """
    Runs a NoonianAgent in a loop, reading from text_snippet_queue and writing responses to llm_response_queue.
    Exits cleanly when stop_flag is set.
    """
    import asyncio

    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    agent = NoonianAgent(args)

    try:
        while not stop_flag.is_set():
            try:
                item = text_snippet_queue.get(timeout=0.1)
            except Exception:
                continue  # Timeout, check stop_flag again

            # item is expected to be (segments, info) or just text
            if isinstance(item, tuple) and len(item) > 0:
                # If using STT, item[0] is a list of segments, each with a .text attribute
                segments = item[0]
                if isinstance(segments, list) and len(segments) > 0 and hasattr(segments[0], "text"):
                    user_input = " ".join([s.text for s in segments])
                else:
                    user_input = str(segments)
            else:
                user_input = str(item)

            if not user_input.strip():
                continue

            # Run the async handle_llm_query in this thread's event loop
            result = loop.run_until_complete(agent.handle_llm_query(user_input))
            last_message = result["messages"][-1]
            mqtt_client.publish(
                args.topic,
                json.dumps({
                    "content": last_message.content,
                    "end": True,
                })
            )
    except Exception as e:
        logger.error(f"noonian_agent_runner encountered an error: {e}")
    finally:
        loop.close()


def mqtt_listener(args, stop_event, text_snippet_queue):
    """Listen on MQTT for text messages and push into TTS queue."""

    def on_message(client, userdata, msg):
        logger.info(f"Got message: {msg}")
        try:
            text = msg.payload.decode("utf-8").strip()
            if text:
                text_snippet_queue.put(text)
        except Exception as e:
            logger.error(f"Failed to handle MQTT message: {e}")

    mqtt_client = mqtt.Client()
    mqtt_client.username_pw_set(username="backbone",password="backbone")
    mqtt_client.on_message = on_message
    mqtt_client.connect(args.broker, args.port, 60)
    mqtt_client.subscribe(args.topic)
    mqtt_client.loop_start()

    while not stop_event.is_set():
        signal.pause()
    mqtt_client.loop_stop()


async def main_async():
    parser = argparse.ArgumentParser(description="Noonian Agent - LLM assistant with tools")
    # LLM configuration
    parser.add_argument("--ollama-model", dest="ollama_model", type=str, default="qwen3:8b",
                       help="Ollama model to use (default: qwen3:30b)")
    parser.add_argument("--ollama-system-prompt", dest="ollama_system_prompt", type=str, 
                       default="You are a helpful assistant.",
                       help="System prompt for the Ollama model")

    # Tool configuration
    parser.add_argument("--tool-file", action="append", default=[],
                       help="Python file implementing an MCP tool")
    parser.add_argument("--tool-server", action="append", default=[],
                       help="URL of an MCP tool server")

    # Parse arguments
    args = parser.parse_args()

    # Set up logging based on verbosity
    if args.tool_file or args.tool_server:
        logger.info(f"Initializing with {len(args.tool_file)} file tools and {len(args.tool_server)} server tools")
    
    stop_event = threading.Event()
    text_snippet_queue = queue.Queue()

    # --- MQTT client ---
    def on_message(client, userdata, msg):
        logger.info(f"Got message: {msg}")
        try:
            text = msg.payload.decode("utf-8").strip()
            if text:
                text_snippet_queue.put(text)
        except Exception as e:
            logger.error(f"Failed to handle MQTT message: {e}")

    mqtt_client = mqtt.Client()
    mqtt_client.username_pw_set(username="backbone",password="backbone")
    mqtt_client.on_message = on_message
    mqtt_client.connect(args.broker, args.port, 60)
    mqtt_client.subscribe(args.topic)
    mqtt_client.loop_start()

    # start Noonian agent worker
    worker = threading.Thread(
        target=noonian_agent_runner,
        args=(args, stop_event, text_snippet_queue, mqtt_client),
        daemon=True,
    )
    worker.start()

    while not stop_event.is_set():
       signal.pause()
    mqtt_client.loop_stop()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level to DEBUG
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    asyncio.run(main_async())
