# Installation

TTS requires python <= 3.12, use `pyenv` to create a virtual environment with a specific python version:
```
# Setup virtual environment with pyenv
pyenv install 3.11.12
pyenv virtualenv 3.11 noonian_venv
pyenv activate noonian_venv

# Install noonian requirements
pip install -r requirements.txt
```

Note that extra steps may be required to make sure `torch` is cuda enabled.

## Ctranslate

If you see an error about ctranslate, look here:
https://github.com/m-bain/whisperX/issues/1038

We force version 4.4.0 of ctranslate so find the library and use patchelf to fix it:
```
# Get path for the library
find $HOME -iname 'libctranslate2-d3638643.so.4.4.0'

# Patch it!
patchelf --clear-execstack PATH_TO/libctranslate2-d3638643.so.4.4.0
```

# Running
```
cd noonian/
python3 -m Noonian.noonian
```