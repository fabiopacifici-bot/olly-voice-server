# Olly Voice Server — production Dockerfile
# GPU-enabled: requires NVIDIA Container Toolkit on the host
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip python3.11-dev \
    ffmpeg libsndfile1 git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python venv
RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Install CosyVoice dependencies from vendor
COPY vendor/CosyVoice/requirements.txt /tmp/cosyvoice-requirements.txt
RUN pip install --no-cache-dir -r /tmp/cosyvoice-requirements.txt || true

# Install Matcha-TTS if it has a setup.py
COPY vendor/CosyVoice/third_party /app/vendor/CosyVoice/third_party
RUN if [ -f vendor/CosyVoice/third_party/Matcha-TTS/setup.py ] || [ -f vendor/CosyVoice/third_party/Matcha-TTS/pyproject.toml ]; then \
      pip install --no-cache-dir -e vendor/CosyVoice/third_party/Matcha-TTS || true; \
    fi

# Copy app code
COPY olly_voice/ /app/olly_voice/
COPY vendor/CosyVoice/cosyvoice/ /app/vendor/CosyVoice/cosyvoice/
COPY ui/ /app/ui/
COPY assets/ /app/assets/
COPY pytest.ini /app/

# Download CosyVoice2-0.5B model at build time
# Override with --build-arg SKIP_MODEL_DOWNLOAD=1 to skip (use volume mount instead)
ARG SKIP_MODEL_DOWNLOAD=0
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "0" ]; then \
      pip install huggingface_hub && \
      python3 -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='/app/vendor/CosyVoice/pretrained_models/CosyVoice2-0.5B')"; \
    fi

# Config via env
ENV LLM_URL=http://ai-server:8005
ENV LLM_MODEL=Qwen/Qwen3-0.6B
ENV WHISPER_MODEL=base.en
ENV WHISPER_DEVICE=cuda
ENV TTS_SPEAKER=default

EXPOSE 8766
CMD ["uvicorn", "olly_voice.app:app", "--host", "0.0.0.0", "--port", "8766"]
