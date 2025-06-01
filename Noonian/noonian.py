import argparse
from queue import Queue
import threading

from . import ShutdownQueue
from .stt import handle_audio_stream, handle_snippet_transcription
from .agent import handle_llm_queries
from .tts import handle_llm_response



class Noonian:
    def __init__(self):
        self.vad_snippet_queue = ShutdownQueue()
        self.text_snippet_queue = ShutdownQueue()
        self.response_queue = ShutdownQueue()

        self.stop_event = threading.Event()

        self.vad_thread = None

    def start(self):
        self.stop_event.clear()
        self.vad_thread = threading.Thread(target=handle_audio_stream, args=(self.stop_event, self.vad_snippet_queue))
        self.vad_thread.start()

        self.transcriber_thread = threading.Thread(target=handle_snippet_transcription, args=(self.stop_event, self.vad_snippet_queue, self.text_snippet_queue))
        self.transcriber_thread.start()

        self.llm_query_thread = threading.Thread(target=handle_llm_queries, args=(self.stop_event, self.text_snippet_queue, self.response_queue))
        self.llm_query_thread.start()

        self.tts_thread = threading.Thread(target=handle_llm_response, args=(self.stop_event, self.response_queue))
        self.tts_thread.start()

    def stop(self):
        self.stop_event.set()
        self.vad_snippet_queue.shutdown()
        self.text_snippet_queue.shutdown()
        self.response_queue.shutdown()

        if self.vad_thread is not None:
            self.vad_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='STT-LLM-TTS')
    args = vars(parser.parse_args())

    n = Noonian()
    n.start()
    input("Press Enter to continue...")
    n.stop()
