# olly-voice-server SKILL

This skill installs a lightweight FastAPI server exposing two endpoints:
- POST /tts with form field `text` -> returns WAV
- POST /transcribe with multipart file `file` -> returns JSON {text}

Setup
1. Place your reference voice at assets/reference_voice.wav
2. Option A: Add CosyVoice as a git submodule under vendor/CosyVoice:
   git submodule add <CosyVoice repo url> vendor/CosyVoice
   cd vendor/CosyVoice && pip install -r requirements.txt
3. Install faster-whisper (recommended): pip install faster-whisper
4. Run: ./setup.sh
5. Run server: uvicorn olly_voice.app:app --host 0.0.0.0 --port 8766

Notes
- Models are lazy-loaded on first request to reduce startup time.
- Do not commit your reference audio or model weights to the repo.
