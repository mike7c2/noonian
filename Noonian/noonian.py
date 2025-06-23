import argparse
import threading
import logging

from . import ShutdownQueue
from .stt import handle_audio_stream, handle_snippet_transcription
from .agent import noonian_agent_runner
from .tts import handle_llm_response

from .logger_config import setup_logger
logger = setup_logger(__name__)

DEFAULT_PROMPT="""
You are Jeeves, a helpful butler!

Your input comes from speech to text, and your output will be expressed via text to speech.

Keep this in mind both when understanding a users input and providing output to them.

Try to be reasonably terse. Don't give extra suggestions in replies unless there is no other content.

You should generally action a users request and respond in a single conversational sentence. Longer replies are acceptable when the user has directly asked for them.

Never output emojis. Do not format lists or write markdown.

If you are not able to do something, escalate it
"""

class Noonian:
    def __init__(self, args):
        self.args = args
        self.vad_snippet_queue = ShutdownQueue()
        self.text_snippet_queue = ShutdownQueue()
        self.response_queue = ShutdownQueue()

        self.stop_event = threading.Event()

        self.vad_thread = None

    def start(self):
        self.stop_event.clear()
        if not self.args.stt_disable:
            logging.info("Starting STT")
            self.vad_thread = threading.Thread(target=handle_audio_stream, args=(self.args, self.stop_event, self.vad_snippet_queue))
            self.vad_thread.start()

            self.transcriber_thread = threading.Thread(target=handle_snippet_transcription, args=(self.args, self.stop_event, self.vad_snippet_queue, self.text_snippet_queue))
            self.transcriber_thread.start()

        if not self.args.ollama_disable:
            logging.info("Starting Agent")
            self.llm_query_thread = threading.Thread(target=noonian_agent_runner, args=(self.args, self.stop_event, self.text_snippet_queue, self.response_queue))
            self.llm_query_thread.start()

        if not self.args.tts_disable:
            logging.info("Starting TTS")
            self.tts_thread = threading.Thread(target=handle_llm_response, args=(self.args, self.stop_event, self.response_queue))
            self.tts_thread.start()

    def stop(self):
        self.stop_event.set()
        self.vad_snippet_queue.shutdown()
        self.text_snippet_queue.shutdown()
        self.response_queue.shutdown()

        if self.vad_thread is not None:
            self.vad_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Noonian is an application running a STT-LLM-TTS loop')
    parser.add_argument("--stt-disable", default=False, action="store_true", help="Disable STT")
    parser.add_argument("--vad-on-threshold", default=0.7, type=float, help="")
    parser.add_argument("--vad-off-threshold", default=0.01, type=float, help="")
    parser.add_argument("--vad-silence-chunks", default=10, type=int, help="")
    parser.add_argument("--vad-repo", default="snakers4/silero-vad", help="")
    parser.add_argument("--vad-model", default="silero_vad", help="")

    parser.add_argument("--whisper-model", default="large-v3-turbo", help="Whisper model")
    parser.add_argument("--whisper-device", default="cuda", help="Whisper model")

    parser.add_argument("--ollama-disable", default=False, action="store_true", help="Disable Ollama")
    parser.add_argument("--ollama-model", default="qwen3:8b", help="Ollama model")
    parser.add_argument("--ollama-system-prompt", default=DEFAULT_PROMPT, help="System prompt for Ollama")
    parser.add_argument("--tool-file", action="append", default=[],
                       help="Python file implementing an MCP tool")
    parser.add_argument("--tool-server", action="append", default=[],
                       help="URL of an MCP tool server")
    parser.add_argument("--tts-disable", default=False, action="store_true", help="Disable TTS")
    parser.add_argument("--tts-model", default="tts_models/multilingual/multi-dataset/xtts_v2", help="Model (v3_en, ....)")
    parser.add_argument("--tts-speaker", default=None, help="Speaker (....)")
    parser.add_argument("--tts-language", default="en", help="TTS Language")
    parser.add_argument("--tts-speaker-wav", default=None, help="TTS Speaker")

    parser.add_argument("--disable-tools", action="store_true", help="Disable tool calls")

    args = parser.parse_args()
    print(args)
    if args.tts_speaker is None and args.tts_speaker_wav is None:
        print("Need at least one of --tts-speaker, --tts-speaker-wav")

    n = Noonian(args)
    n.start()
    input("Press Enter to continue...")
    n.stop()
