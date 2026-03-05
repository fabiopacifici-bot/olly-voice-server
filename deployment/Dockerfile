FROM python:3.11-slim

WORKDIR /app

# System deps for audio
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY olly_voice/ ./olly_voice/
COPY ui/ ./ui/

EXPOSE 8766

CMD ["uvicorn", "olly_voice.app:app", "--host", "0.0.0.0", "--port", "8766"]
