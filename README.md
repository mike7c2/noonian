# Installation

```
pip install -r requirements.txt
```

Note that extra steps may be required to make sure `torch` is cuda enabled.

## TTS

Text to speech relies on a Coqui TTS server which can be run with docker: https://docs.coqui.ai/en/latest/docker_images.html

On windows, docker desktop for windows should support GPU acceleration immediately

On linux, make sure that the nvidia container toolkit is installed: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Use the following command to launch a server with a model.etc:
```
docker run --rm -it -p 5002:5002 --gpus all --entrypoint /bin/bash ghcr.io/coqui-ai/tts -c "python3 TTS/server/server.py --model_name tts_models/en/vctk/vits --use_cuda true"
```

WIP

# Running
```
cd noonian/
python3 -m Noonian.noonian
```