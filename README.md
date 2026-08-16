# Olly Voice Server

[![Hippocratic License HL3-LAW-MIL-SV](https://img.shields.io/static/v1?label=Hippocratic%20License&message=HL3-LAW-MIL-SV&labelColor=5e2751&color=bc8c3d)](https://firstdonoharm.dev/version/3/0/law-mil-sv.html)

Local voice assistant: mic → STT → LLM → TTS → speaker

Prerequisites
- NVIDIA GPU with NVIDIA Container Toolkit (for GPU Docker)
- Docker & docker-compose OR Python 3.11+ for local dev

Docker quick-start

# Start private-ai-server first (on host/Windows side):
# cd D:\ai-server-py
# python -m uvicorn app:app --host 0.0.0.0 --port 8005

# Build and run voice server:
cd repositories/olly-voice-server
docker build -t olly-voice .
docker run --gpus all -p 8766:8766 -e LLM_URL=http://host.docker.internal:8005 olly-voice

# Open UI:
# http://localhost:8766/ui/index.html

Non-Docker (local dev)

# Create venv, install deps and run
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
uvicorn olly_voice.app:app --host 0.0.0.0 --port 8766

API endpoints
- GET /health
- POST /transcribe
- POST /tts
- POST /chat

Environment variables (selected)
- LLM_URL: URL of private ai-server (default: http://ai-server:8005)
- LLM_MODEL: Model id used by ai-server
- WHISPER_MODEL: Whisper model (default: base.en)
- WHISPER_DEVICE: cpu|cuda
- TTS_SPEAKER: default speaker id

Notes
- Model weights are large and should NOT be committed. The folder vendor/CosyVoice/pretrained_models/ is ignored by .gitignore.
- To skip downloading CosyVoice models at Docker build time, pass --build-arg SKIP_MODEL_DOWNLOAD=1 and mount models into the cosyvoice-models volume.
