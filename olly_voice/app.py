from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse
import os
from pathlib import Path
import asyncio
import tempfile
import uuid

app = FastAPI()
BASE = Path(__file__).resolve().parent.parent
ASSETS = BASE / "assets"
MODELS = BASE / "models"

# Lazy-loaded globals
_cosy = None
_whisper = None

# NOTE: _cosy and _whisper are lazy-load stubs for now. They are placeholders
# until CosyVoice and the WhisperModel are wired in. Do NOT attempt to import
# or initialize real models here in this change.

async def load_cosy():
    global _cosy
    if _cosy is None:
        # Placeholder for CosyVoice lazy load
        _cosy = "cosy-loaded"
    return _cosy

async def load_whisper():
    global _whisper
    if _whisper is None:
        # Placeholder for faster-whisper lazy load
        _whisper = "whisper-loaded"
    return _whisper

@app.post("/tts")
async def tts(text: str = Form(...)):
    await load_cosy()
    # Use a per-request temporary file to avoid concurrent-writes to a fixed path
    fd, path = tempfile.mkstemp(suffix=".wav", dir=str(ASSETS))
    os.close(fd)
    try:
        # Create a 1-second silent wav as placeholder
        import wave
        import struct
        framerate = 16000
        duration = 1
        nframes = duration * framerate
        with wave.open(path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            for i in range(nframes):
                wf.writeframes(struct.pack('<h', 0))
        # Stream file back to client and remove after streaming
        def iterfile(p):
            with open(p, 'rb') as fh:
                while True:
                    chunk = fh.read(8192)
                    if not chunk:
                        break
                    yield chunk
        return StreamingResponse(iterfile(path), media_type='audio/wav')
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    await load_whisper()
    # Save uploaded to a per-request temporary file to avoid races
    fd, path = tempfile.mkstemp(suffix=".wav", dir=str(ASSETS))
    os.close(fd)
    try:
        with open(path, 'wb') as f:
            f.write(await file.read())
        # Placeholder transcription
        return {"text": "[transcription placeholder]"}
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
