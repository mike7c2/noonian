import pyaudio
import threading
import io
import wave
import re
import nltk
import numpy as np
from TTS.api import TTS
from . import QueueShutdownException

from .logger_config import setup_logger
logger = setup_logger(__name__)

def clean_string(text):
    # Allows letters, numbers, whitespace, and basic punctuation
    return re.sub(r"[a-zA-Z0-9\s.,!?;:'\"()\-_/]", "", text)

def audio_streamer(args, audio, wf):
    logger.info(f'Starting playback')
    stream = audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=24000,
        output=True
    )
    stream.write(wf.tobytes())
    stream.close()

def handle_llm_response(args, stop_event, llm_response_queue):
    try:
        audio = pyaudio.PyAudio()
        tts_instance = TTS(model_name=args.tts_model).to("cuda")
        player_thread = None

        input_buffer = ""
        while not stop_event.is_set():
            # Add any new content to the input buffer
            response = llm_response_queue.get()
            if response["content"] is not None:
                input_buffer += response["content"]
            flush = response["end"]

            # Determine if there is text ready for TTS
            preprocessed_lines = nltk.sent_tokenize(input_buffer)
            if flush: # On a flush send all lines, empty the buffer
                input_buffer = ""
            elif len(preprocessed_lines) > 1: # If we have an entire sentence ready, send it
                input_buffer = preprocessed_lines[-1]
                preprocessed_lines = preprocessed_lines[:-1]
            else:
                continue

            # Tidy up lines
            preprocessed_lines = [x.strip() for x in preprocessed_lines if len(x.strip()) > 0]

            for i, line in enumerate(preprocessed_lines):
                logger.info(f'Processing line {i+1}/{len(preprocessed_lines)}: {line}')
                try:
                    wf = tts_instance.tts(line, language=args.tts_language, speaker_wav=args.tts_speaker_wav)
                    wf = np.array(wf, dtype=np.float32)

                    # Don't play more audio if some is already playing, wait for the last snippet to complete
                    if player_thread is not None:
                        player_thread.join()
                        player_thread = None

                    # Start player thread (note, doing this in a thread means we can start TTS on the next
                    # block whilst the current block is playing)                    
                    player_thread = threading.Thread(target=audio_streamer, args=(args, audio, wf))
                    player_thread.start()
                except ValueError as e:
                    logger.warning(f'TTS failed for line: {line}. Error: {str(e)}. Skipping...')
    except QueueShutdownException:
        pass
