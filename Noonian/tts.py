import numpy as np
from silero_tts.silero_tts import SileroTTS
import wave
import pyaudio
import threading
from . import QueueShutdownException

def audio_streamer(audio, samples):
    stream = audio.open(format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True)
    stream.write(samples.tobytes())
    stream.close()

def handle_llm_response(stop_event, llm_response_queue):
    try:
        audio = pyaudio.PyAudio()    


        tts = SileroTTS(model_id='v3_en', language='en', speaker='en_2', sample_rate=24000, device='cpu')

        while not stop_event.is_set():
            
            response = llm_response_queue.get()
            response_content = response["message"]["content"]
            print(response_content)
            preprocessed_lines = tts.preprocess_text(response_content)
            player_thread = None

            for i, line in enumerate(preprocessed_lines):
                print(f'Processing line {i+1}/{len(preprocessed_lines)}: {line}')
                try:
                    
                    gen_audio = tts.tts_model.apply_tts(text=line,
                                                    speaker=tts.speaker,
                                                    sample_rate=tts.sample_rate,
                                                    put_accent=tts.put_accent,
                                                    put_yo=tts.put_yo)
                    samples = (gen_audio * 32767).numpy().astype('int16')
                    if player_thread is not None:
                        player_thread.join()
                    player_thread = threading.Thread(target=audio_streamer, args=(audio, samples))
                    player_thread.start()


                except ValueError as e:
                    print(f'TTS failed for line: {line}. Error: {str(e)}. Skipping...')


    except QueueShutdownException:
        pass