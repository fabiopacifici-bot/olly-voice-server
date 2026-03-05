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
import asyncio

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

# In-memory conversation store: session_id -> list of {'role': 'user'|'assistant'|'system', 'content': '...'}
_conversations = {}
_MAX_TURNS = 20
_DEFAULT_SYSTEM_PROMPT = "You are Olly, a helpful and concise voice assistant. Keep your answers short and conversational — 1-3 sentences max. No bullet lists, no markdown, no internal reasoning, just natural speech. /no_think"

# Lazy-loaded globals
_cosy = None
_whisper = None
_tts_available = False
_cosy_load_attempted = False
_whisper_load_attempted = False

# --- Whisper loader ---
def load_whisper():
    global _whisper, _whisper_load_attempted
    if _whisper_load_attempted:
        return _whisper
    _whisper_load_attempted = True
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

# --- Piper TTS loader ---
PIPER_MODEL = os.environ.get('PIPER_MODEL', str(BASE / 'pretrained_models' / 'piper' / 'en_US-amy-medium.onnx'))

def load_cosy():
    global _cosy, _tts_available, _cosy_load_attempted
    if _cosy_load_attempted:
        return _cosy
    _cosy_load_attempted = True
    try:
        from piper import PiperVoice
        import wave, io
        voice = PiperVoice.load(PIPER_MODEL)
        def inference_sft(text, **kwargs):
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                voice.synthesize_wav(text, wf)
            return buf.getvalue()
        _cosy = {'inference_sft': inference_sft}
        _tts_available = True
        print(f"Piper TTS loaded: {PIPER_MODEL}")
        return _cosy
    except Exception as e:
        print("Piper TTS load failed:", e)
        _cosy = None
        _tts_available = False
        return None

@app.on_event("startup")
async def startup_event():
    # check LLM reachability (warn only)
    try:
        async with httpx.AsyncClient(timeout=1.0) as c:
            await c.get(LLM_URL)
    except Exception:
        print(f"WARNING: LLM server at {LLM_URL} appears unreachable.")
    # load models in background threads so handlers don't block
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_whisper)
    await loop.run_in_executor(None, load_cosy)

@app.get('/health')
async def health():
    status = {'status': 'ok'}
    # report whisper/tts availability based on already-loaded globals
    status['whisper'] = 'ok' if _whisper is not None else 'missing'
    status['tts'] = 'piper' if _tts_available else 'stub'
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

# Conversation history helpers
def _get_history(session_id: str):
    return _conversations.get(session_id, [])

def _append_turn(session_id: str, role: str, content: str):
    hist = _conversations.setdefault(session_id, [])
    hist.append({'role': role, 'content': content})
    # Keep only last _MAX_TURNS * 2 messages (user+assistant counted as 1 turn each)
    if len(hist) > _MAX_TURNS * 2:
        # drop oldest
        _conversations[session_id] = hist[-(_MAX_TURNS*2):]

@app.get('/history/{session_id}')
async def get_history(session_id: str):
    return JSONResponse({'session_id': session_id, 'history': _get_history(session_id)})

@app.delete('/history/{session_id}')
async def delete_history(session_id: str):
    _conversations.pop(session_id, None)
    return JSONResponse({'status': 'cleared', 'session_id': session_id})

# /tts endpoint: accepts form text and returns wav stream
@app.post('/tts')
async def tts(text: str = Form(...)):
    # Do not call load_cosy here; startup should have loaded models. If not loaded, attempt non-blocking check.
    tmpdir = tempfile.gettempdir()
    fd, path = tempfile.mkstemp(suffix='.wav', dir=tmpdir)
    os.close(fd)
    try:
        if _tts_available and _cosy:
            loop = asyncio.get_event_loop()
            try:
                wav_bytes = await loop.run_in_executor(None, lambda: _cosy['inference_sft'](text, speaker=TTS_SPEAKER))
            except Exception:
                wav_bytes = await loop.run_in_executor(None, lambda: _cosy['inference_zero_shot'](text, speaker=TTS_SPEAKER))
            if isinstance(wav_bytes, bytes):
                with open(path, 'wb') as f:
                    f.write(wav_bytes)
            elif isinstance(wav_bytes, str) and os.path.exists(wav_bytes):
                # copy file
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
        return StreamingResponse(iterfile(path), media_type='audio/wav', background=BackgroundTask(os.remove, path))
    except Exception as e:
        try:
            os.remove(path)
        except Exception:
            pass
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post('/transcribe')
async def transcribe(file: UploadFile = File(...)):
    model = _whisper
    tmpdir = tempfile.gettempdir()
    fd, path = tempfile.mkstemp(suffix='.wav', dir=tmpdir)
    os.close(fd)
    try:
        with open(path,'wb') as f:
            f.write(await file.read())
        if model is None:
            return {'text': ''}
        # Use faster-whisper to transcribe in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(None, lambda: list(model.transcribe(path, beam_size=5)))
        text = ''
        for segment in segments:
            text += segment.text
        return {'text': text}
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

# /chat: accept JSON {text:..., session_id:..., system_prompt:...} → call LLM → TTS → return wav + headers
@app.post('/chat')
async def chat(body: dict):
    import time as _time
    _t0 = _time.time()
    if not isinstance(body, dict):
        return JSONResponse({'error': 'invalid body'}, status_code=400)
    user_text = body.get('text')
    session_id = body.get('session_id', 'default')
    system_prompt = body.get('system_prompt') or _DEFAULT_SYSTEM_PROMPT
    if not user_text:
        return JSONResponse({'error': 'missing text'}, status_code=400)
    # Append user turn
    _append_turn(session_id, 'user', user_text)
    # Prepare messages for LLM: system prompt + history
    history = _get_history(session_id)
    messages = [{'role': 'system', 'content': system_prompt}]
    for m in history:
        messages.append({'role': m['role'], 'content': m['content']})
    # Call LLM
    payload = {'messages': messages, 'model': LLM_MODEL, 'max_tokens': 200}
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(f"{LLM_URL}/chat", json=payload)
            r.raise_for_status()
            jr = r.json()
            print(f"[TIMING] LLM: {(_time.time()-_t0)*1000:.0f}ms")
            if isinstance(jr, dict) and 'choices' in jr:
                reply = jr['choices'][0].get('message', {}).get('content') or jr['choices'][0].get('content') or ''
            elif isinstance(jr, dict) and 'message' in jr:
                msg = jr['message']
                reply = msg.get('content', '') if isinstance(msg, dict) else str(msg)
            else:
                reply = jr.get('text', '') if isinstance(jr, dict) else str(jr)
    except Exception as e:
        return JSONResponse({'error': f'LLM call failed: {e}'}, status_code=502)
    _t1 = _time.time()
    # Strip Qwen3 <think>...</think> reasoning blocks before TTS
    import re
    reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
    # Truncate at first "User:" continuation hallucination
    for stop in ['\nUser:', '\nuser:', '\nAssistant:', '\nassistant:', '\n\n\n']:
        if stop in reply:
            reply = reply[:reply.index(stop)].strip()
    # Append assistant reply to history
    _append_turn(session_id, 'assistant', reply)
    # Synthesize via TTS
    tmpdir = tempfile.gettempdir()
    fd, path = tempfile.mkstemp(suffix='.wav', dir=tmpdir)
    os.close(fd)
    try:
        if _tts_available and _cosy:
            loop = asyncio.get_event_loop()
            try:
                wav_bytes = await loop.run_in_executor(None, lambda: _cosy['inference_sft'](reply, speaker=TTS_SPEAKER))
            except Exception:
                wav_bytes = await loop.run_in_executor(None, lambda: _cosy['inference_zero_shot'](reply, speaker=TTS_SPEAKER))
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
        print(f"[TIMING] TTS/WAV: {(_time.time()-_t1)*1000:.0f}ms")
        def iterfile(p):
            with open(p,'rb') as fh:
                while True:
                    chunk = fh.read(8192)
                    if not chunk:
                        break
                    yield chunk
        # Sanitize reply for HTTP header (strip newlines/non-ASCII)
        safe_reply = reply.replace('\n', ' ').replace('\r', '').encode('ascii', 'ignore').decode('ascii')[:500]
        headers = {'X-Assistant-Reply': safe_reply, 'X-Session-Id': session_id}
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
