#!/usr/bin/env python3
import argparse
import signal
import sys
import logging
import threading
import queue
from collections import deque
import numpy as np
import sounddevice as sd
import paho.mqtt.client as mqtt
from voxcpm import VoxCPM

logger = logging.getLogger("tts-mqtt-voxcpm")

# --- Defaults ---
DEFAULT_TOPIC = "llm-response"
DEFAULT_BROKER = "localhost"
DEFAULT_PORT = 1883

STREAMING = False


def speak(args, model, text: str, buffer_size=100):
    """Generate speech for the given text and play it smoothly in a streaming fashion."""
    chunk_buffer = deque()
    if STREAMING:
        with sd.OutputStream(samplerate=16000, channels=1, dtype='float32') as stream:
            for chunk in model.generate_streaming(
                text=text,
                prompt_wav_path=args.reference_wav,
                prompt_text=args.reference_text,
                cfg_value=2.0,
                inference_timesteps=10,
                normalize=True,
                denoise=False
            ):
                # Convert chunk to float32
                chunk = np.array(chunk, dtype='float32')
                chunk_buffer.append(chunk)
                
                # If we have enough buffered chunks, write them together
                if len(chunk_buffer) >= buffer_size:
                    stream.write(np.concatenate(chunk_buffer))
                    chunk_buffer.clear()
            
            # Write any remaining chunks
            if chunk_buffer:
                stream.write(np.concatenate(chunk_buffer))
    else:
        audio = model.generate(
            text=text,
            prompt_wav_path=args.reference_wav,
            prompt_text=args.reference_text,
            cfg_value=2.0,
            inference_timesteps=10,
            normalize=True,
            denoise=False
        )

        # Convert to float32 NumPy array if not already
        audio = np.array(audio, dtype='float32')

        # Play it back in real time using sounddevice
        with sd.OutputStream(samplerate=16000, channels=1, dtype='float32') as stream:
            stream.write(audio)


def tts_worker(args, stop_event, tts_queue):

    model = VoxCPM.from_pretrained(
        hf_model_id="openbmb/VoxCPM-0.5B",
        zipenhancer_model_id="iic/speech_zipenhancer_ans_multiloss_16k_base"
    )
    while not stop_event.is_set():
        try:
            text = tts_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        logger.info(f"TTS speaking: {text}")
        try:
            speak(args, model, text)
        except Exception as e:
            logger.error(f"TTS failed for '{text}': {e}")


def mqtt_listener(args, stop_event, tts_queue):
    """Listen on MQTT for text messages and push into TTS queue."""

    def on_message(client, userdata, msg):
        logger.info(f"Got message: {msg}")
        try:
            text = msg.payload.decode("utf-8").strip()
            if text:
                tts_queue.put(text)
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


def main():
    parser = argparse.ArgumentParser(description="VoxCPM TTS MQTT consumer and audio player")
    parser.add_argument("--reference-wav", default=None, help="The default wavfile to use as a style reference")
    parser.add_argument("--reference-text", default=None, help="Text of the speech in the reference wavfile")
    parser.add_argument("--broker", default=DEFAULT_BROKER, help="MQTT broker address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="MQTT broker port")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="MQTT topic to subscribe")
    parser.add_argument("--model", default="openbmb/VoxCPM-0.5B", help="VoxCPM model to load")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # --- Logging ---
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info(f"Loading VoxCPM model '{args.model}'...")
    stop_event = threading.Event()
    tts_queue = queue.Queue()

    # Start TTS worker thread
    worker = threading.Thread(target=tts_worker, args=(args, stop_event, tts_queue), daemon=True)
    worker.start()

    def handle_sigint(sig, frame):
        logger.info("Shutting down...")
        stop_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # Start MQTT listener (blocking)
    mqtt_listener(args, stop_event, tts_queue)


if __name__ == "__main__":
    main()
