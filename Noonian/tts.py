import pyaudio
import threading
import requests
import io
import wave
import re
import nltk

from . import QueueShutdownException

from .logger_config import setup_logger
logger = setup_logger(__name__)

def clean_string(text):
    # Allows letters, numbers, whitespace, and basic punctuation
    return re.sub(r"[a-zA-Z0-9\s.,!?;:'\"()\-_/]", "", text)

def audio_streamer(args, audio, wf):
    logger.info(f'Starting playback')
    stream = audio.open(
        format=wf.getsampwidth(),
        channels=wf.getnchannels(),
        rate=wf.getframerate()//2,
        output=True
    )
    stream.write(wf.readframes(wf.getnframes()))
    stream.close()

def handle_llm_response(args, stop_event, llm_response_queue):
    try:
        audio = pyaudio.PyAudio()
        player_thread = None

        while not stop_event.is_set():
            
            response = llm_response_queue.get()
            response_content = response
            logger.debug(response_content)
            response_content = response_content.split("</think>")[-1]
            preprocessed_lines = nltk.sent_tokenize(response_content)
            preprocessed_lines = [x.strip() for x in preprocessed_lines if len(x.strip()) > 0]

            for i, line in enumerate(preprocessed_lines):
                logger.info(f'Processing line {i+1}/{len(preprocessed_lines)}: {line}')
                try:
                    headers = {
                        "text": line,
                    }
                    if args.tts_speaker_wav is not None:
                        headers["style-wav"] = args.tts_speaker_wav
                    if args.tts_speaker is not None:
                        headers["speaker-id"] = args.tts_speaker
                    if args.tts_language is not None:
                        headers["language-id"] = args.tts_language
                    logger.info(headers)
                    response = requests.post(args.tts_url, headers=headers)

                    if response.status_code == 200:
                        wav_io = io.BytesIO(response.content)
                        wf = wave.open(wav_io, 'rb')
                        if player_thread is not None:
                            player_thread.join()
                            player_thread = None
                        player_thread = threading.Thread(target=audio_streamer, args=(args, audio, wf))
                        player_thread.start()
                    else:
                        logger.error(f"Request failed with status code {response.status_code}")
                        logger.error(response.text)


                except ValueError as e:
                    logger.warning(f'TTS failed for line: {line}. Error: {str(e)}. Skipping...')


    except QueueShutdownException:
        pass
