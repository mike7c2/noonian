#!/usr/bin/env python3
import pyaudio
import numpy as np
import torch
import whisper
import logging
import threading
import signal
import sys
import argparse
import paho.mqtt.client as mqtt
import time
import queue

logger = logging.getLogger("stt-mqtt")

# --- Audio settings ---
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 100)

# --- MQTT defaults ---
DEFAULT_TOPIC = "stt-message"
DEFAULT_BROKER = "localhost"
DEFAULT_PORT = 1883


def int2float(sound):
    sound = sound.astype("float32")
    sound *= 1 / 32768
    return sound


def init_audio_stream():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )
    return stream


def audio_loop(args, stop_event, snippet_queue):
    """Handles audio stream and voice activity detection."""
    model, utils = torch.hub.load(
        repo_or_dir=args.vad_repo,
        model=args.vad_model,
        force_reload=True,
    )
    stream = init_audio_stream()
    snippet_active = False
    snippet_chunks = []
    off_cnt = 0

    logger.info("Audio/VAD loop started")
    while not stop_event.is_set():
        audio_chunk = stream.read(512, exception_on_overflow=False)
        audio_int16 = np.frombuffer(audio_chunk, np.int16)
        audio_float32 = int2float(audio_int16)

        # --- VAD timing ---
        if args.verbose:
            t0 = time.perf_counter()
        new_confidence = model(torch.from_numpy(audio_float32), 16000).item()
        if args.verbose:
            dt = (time.perf_counter() - t0) * 1000
            logger.debug(f"VAD inference took {dt:.2f} ms, confidence={new_confidence:.3f}")

        if snippet_active and new_confidence < args.vad_off_threshold:
            off_cnt += 1
            if off_cnt >= args.vad_silence_chunks:
                logger.info("Ended clip")
                snippet_active = False
                snippet_queue.put(snippet_chunks)  # push to queue
                snippet_chunks = []
                off_cnt = 0
        elif not snippet_active and new_confidence > args.vad_on_threshold:
            logger.info("Started clip")
            snippet_chunks.append(audio_int16)
            snippet_active = True
        elif snippet_active:
            snippet_chunks.append(audio_int16)
            if new_confidence > args.vad_off_threshold:
                off_cnt = 0


def transcription_worker(args, stop_event, snippet_queue, mqtt_client):
    """Background thread: transcribes audio snippets and publishes to MQTT."""
    model = whisper.load_model(args.whisper_model)
    logger.info(f"Loaded Whisper model '{args.whisper_model}'")

    while not stop_event.is_set():
        try:
            snippet_chunks = snippet_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        assembled_snippet = np.zeros(len(snippet_chunks) * 512, dtype=np.float32)
        for i in range(len(snippet_chunks)):
            assembled_snippet[i * 512 : (i + 1) * 512] = snippet_chunks[i]

        logger.info("Processing clip")

        if args.verbose:
            t0 = time.perf_counter()
        result = model.transcribe(
            assembled_snippet,
            beam_size=5,
            language="en",
        )
        if args.verbose:
            dt = (time.perf_counter() - t0) / 1000
            logger.debug(f"Transcription took {dt:.2f} s")

        text = result.get("text", "").strip()
        logger.info(f"Transcription: {text}")

        if text:
            mqtt_client.publish(args.topic, text)


def main():
    parser = argparse.ArgumentParser(description="Speech-to-text MQTT publisher")
    parser.add_argument("--broker", default=DEFAULT_BROKER, help="MQTT broker address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="MQTT broker port")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="MQTT topic to publish")
    parser.add_argument("--vad-repo", default="snakers4/silero-vad", help="TorchHub VAD repo")
    parser.add_argument("--vad-model", default="silero_vad", help="TorchHub VAD model")
    parser.add_argument("--vad-on-threshold", type=float, default=0.6, help="VAD on threshold")
    parser.add_argument("--vad-off-threshold", type=float, default=0.3, help="VAD off threshold")
    parser.add_argument("--vad-silence-chunks", type=int, default=10, help="Silence chunks before ending speech")
    parser.add_argument("--whisper-model", default="turbo", help="Whisper model to use")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    # --- Logging ---
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    # --- MQTT client ---
    mqtt_client = mqtt.Client()
    mqtt_client.username_pw_set(username="backbone",password="backbone")
    mqtt_client.connect(args.broker, args.port, 60, )
    mqtt_client.loop_start()

    stop_event = threading.Event()
    snippet_queue = queue.Queue()

    # start transcription worker
    worker = threading.Thread(
        target=transcription_worker,
        args=(args, stop_event, snippet_queue, mqtt_client),
        daemon=True,
    )
    worker.start()

    def handle_sigint(sig, frame):
        logger.info("Shutting down...")
        stop_event.set()
        mqtt_client.loop_stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    audio_loop(args, stop_event, snippet_queue)


if __name__ == "__main__":
    main()
