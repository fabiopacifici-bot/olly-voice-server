from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from pathlib import Path
import os
import tempfile
import traceback
import httpx
import sys

app = FastAPI()
BASE = Path(__file__).resolve().parent.parent
ASSETS = BASE / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# Config via env
LLM_URL = os.environ.get('LLM_URL', 'http://localhost:8005')
LLM_MODEL = os.environ.get('LLM_MODEL', 'Qwen/Qwen3-0.6B')
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'base.en')
WHISPER_DEVICE = os.environ.get('WHISPER_DEVICE', 'cuda')
TTS_SPEAKER = os.environ.get('TTS_SPEAKER', 'default')

# Lazy-loaded globals
_cosy = None
_whisper = None
_tts_available = False

# Attempt a startup check for LLM (warn only)
try:
    with httpx.Client(timeout=1.0) as c:
        c.get(LLM_URL)
except Exception:
    print(f"WARNING: LLM server at {LLM_URL} appears unreachable. /health will report a warning.")

# --- Whisper loader ---
def load_whisper():
    global _whisper
    if _whisper is not None:
        return _whisper
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type='float16')
        _whisper = model
        return _whisper
    except Exception as e:
        print("Failed to load faster-whisper:", e)
        _whisper = None
        return None

# --- Cosy/TTS loader ---
def load_cosy():
    global _cosy, _tts_available
    if _cosy is not None:
        return _cosy
    try:
        # Add vendor paths so CosyVoice can import its modules
        vendor_dir = BASE / 'vendor'
        sys.path.insert(0, str(vendor_dir))
        # CosyVoice package expected under vendor/CosyVoice
        from CosyVoice.inference import inference_sft, inference_zero_shot
        _cosy = {'inference_sft': inference_sft, 'inference_zero_shot': inference_zero_shot}
        _tts_available = True
        return _cosy
    except Exception as e:
        print("CosyVoice load failed:", e)
        traceback.print_exc()
        _cosy = None
        _tts_available = False
        return None

@app.get('/health')
async def health():
    status = {'status': 'ok'}
    # warn if whisper or cosy not available
    if load_whisper() is None:
        status['whisper'] = 'missing'
    else:
        status['whisper'] = 'ok'
    load_cosy()
    status['tts'] = 'ok' if _tts_available else 'stub'
    # check LLM
    try:
        async with httpx.AsyncClient(timeout=1.0) as c:
            await c.get(LLM_URL)
        status['llm'] = 'ok'
    except Exception:
        status['llm'] = 'unreachable'
    return JSONResponse(status)

# Helper to create a silent wav
def make_silent_wav(path, duration_s=1, rate=16000):
    import wave, struct
    nframes = int(duration_s * rate)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        for _ in range(nframes):
            wf.writeframes(struct.pack('<h', 0))

# /tts endpoint: accepts form text and returns wav stream
@app.post('/tts')
async def tts(text: str = Form(...)):
    load_cosy()
    tmpdir = tempfile.gettempdir()
    fd, path = tempfile.mkstemp(suffix='.wav', dir=tmpdir)
    os.close(fd)
    try:
        if _tts_available and _cosy:
            # Use inference_sft if available, otherwise zero_shot
            try:
                wav_bytes = _cosy['inference_sft'](text, speaker=TTS_SPEAKER)
            except Exception:
                wav_bytes = _cosy['inference_zero_shot'](text, speaker=TTS_SPEAKER)
            # If the inference returned bytes, write them; else, assume path
            if isinstance(wav_bytes, bytes):
                with open(path, 'wb') as f:
                    f.write(wav_bytes)
            elif isinstance(wav_bytes, str) and os.path.exists(wav_bytes):
                # copy file
                with open(wav_bytes, 'rb') as src, open(path, 'wb') as dst:
                    dst.write(src.read())
            else:
                # fallback
                make_silent_wav(path)
        else:
            make_silent_wav(path)

        def iterfile(p):
            with open(p,'rb') as fh:
                while True:
                    chunk = fh.read(8192)
                    if not chunk:
                        break
                    yield chunk
        return StreamingResponse(iterfile(path), media_type='audio/wav', background=BackgroundTask(os.remove, path))
    except Exception as e:
        try:
            os.remove(path)
        except Exception:
            pass
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post('/transcribe')
async def transcribe(file: UploadFile = File(...)):
    model = load_whisper()
    tmpdir = tempfile.gettempdir()
    fd, path = tempfile.mkstemp(suffix='.wav', dir=tmpdir)
    os.close(fd)
    try:
        with open(path,'wb') as f:
            f.write(await file.read())
        if model is None:
            return {'text': ''}
        # Use faster-whisper to transcribe
        segments, info = model.transcribe(path, beam_size=5)
        text = ''
        for segment in segments:
            text += segment.text
        return {'text': text}
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

# /chat: accept JSON {text:...} → call LLM → TTS → return wav
@app.post('/chat')
async def chat(body: dict):
    user_text = body.get('text') if isinstance(body, dict) else None
    if not user_text:
        return JSONResponse({'error': 'missing text'}, status_code=400)
    # Call LLM
    payload = {'messages': [{'role':'user','content': user_text}], 'model': LLM_MODEL}
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(f"{LLM_URL}/chat", json=payload)
            r.raise_for_status()
            jr = r.json()
            # Support multiple response shapes
            if isinstance(jr, dict) and 'choices' in jr:
                reply = jr['choices'][0].get('message', {}).get('content') or jr['choices'][0].get('content') or ''
            else:
                reply = jr.get('text') if isinstance(jr, dict) else str(jr)
    except Exception as e:
        return JSONResponse({'error': f'LLM call failed: {e}'}, status_code=502)
    # Synthesize via TTS
    # Reuse tts() implementation by calling internal function
    tmpdir = tempfile.gettempdir()
    fd, path = tempfile.mkstemp(suffix='.wav', dir=tmpdir)
    os.close(fd)
    try:
        if _tts_available and _cosy:
            try:
                wav_bytes = _cosy['inference_sft'](reply, speaker=TTS_SPEAKER)
            except Exception:
                wav_bytes = _cosy['inference_zero_shot'](reply, speaker=TTS_SPEAKER)
            if isinstance(wav_bytes, bytes):
                with open(path,'wb') as f:
                    f.write(wav_bytes)
            elif isinstance(wav_bytes, str) and os.path.exists(wav_bytes):
                with open(wav_bytes,'rb') as src, open(path,'wb') as dst:
                    dst.write(src.read())
            else:
                make_silent_wav(path)
        else:
            make_silent_wav(path)
        def iterfile(p):
            with open(p,'rb') as fh:
                while True:
                    chunk = fh.read(8192)
                    if not chunk:
                        break
                    yield chunk
        headers = {'X-Assistant-Reply': reply}
        return StreamingResponse(iterfile(path), media_type='audio/wav', background=BackgroundTask(os.remove, path), headers=headers)
    except Exception as e:
        try:
            os.remove(path)
        except Exception:
            pass
        return JSONResponse({'error': str(e)}, status_code=500)

# Serve the UI static files under /ui
ui_path = BASE / 'ui'
if ui_path.exists():
    app.mount('/ui', StaticFiles(directory=str(ui_path)), name='ui')
