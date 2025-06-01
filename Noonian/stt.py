import pyaudio
import threading
import numpy as np
import torch
from faster_whisper import WhisperModel
from . import QueueShutdownException

WHISPER_MODEL = "large-v3-turbo"
SAMPLE_RATE=16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)
CONFIDENCE_ON_THRESHOLD=0.7
CONFIDENCE_OFF_THRESHOLD=0.01
END_DEBOUNCE=10

def init_audio_stream():
    audio = pyaudio.PyAudio()    
    stream = audio.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)
    return stream

def int2float(sound):
    sound = sound.astype('float32')
    sound *= 1/32768
    return sound

def handle_audio_stream(stop_event, vad_snippet_queue):
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=True)
        stream = init_audio_stream()
        snippet_active = False
        snippet_chunks = []
        off_cnt = 0

        while not stop_event.is_set():
            audio_chunk = stream.read(512)
            audio_int16 = np.frombuffer(audio_chunk, np.int16)
            audio_float32 = int2float(audio_int16)
            new_confidence = model(torch.from_numpy(audio_float32), 16000).item()

            if snippet_active and new_confidence < CONFIDENCE_OFF_THRESHOLD:
                off_cnt += 1
                if off_cnt >= END_DEBOUNCE:
                    snippet_active = False
                    vad_snippet_queue.put(snippet_chunks)
                    snippet_chunks = []
                    off_cnt = 0
                else:
                    snippet_chunks.append(audio_int16)
            elif not snippet_active and new_confidence > CONFIDENCE_ON_THRESHOLD:
                snippet_chunks.append(audio_int16)
                snippet_active = True
            elif snippet_active:
                snippet_chunks.append(audio_int16)
                if new_confidence > CONFIDENCE_OFF_THRESHOLD:
                    off_cnt = 0

    except QueueShutdownException:
        pass

def handle_snippet_transcription(stop_event, vad_snippet_queue, text_snippet_queue):
    try:
        model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")

        while not stop_event.is_set():
            snippet = vad_snippet_queue.get()
            assembled_snippet = np.zeros(len(snippet)*512)
            for i in range(len(snippet)):
                assembled_snippet[i*512:(i+1)*512] = snippet[i]
            segments, info = model.transcribe(assembled_snippet, beam_size=5)
            text_snippet_queue.put((segments, info))
    except QueueShutdownException:
        pass