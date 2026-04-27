from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from pathlib import Path
from datetime import datetime as _dt
import os
import tempfile
import traceback
import httpx
import sys
import asyncio

# CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8766", "http://127.0.0.1:8766"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Assistant-Reply", "X-Thinking", "X-Session-Id"],
)

BASE = Path(__file__).resolve().parent.parent
ASSETS = BASE / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
MEETINGS_DIR = Path(os.environ.get('MEETINGS_DIR', Path.home() / 'Documents' / 'Meetings'))
MEETINGS_DIR.mkdir(parents=True, exist_ok=True)

# Config via env
LLM_URL = os.environ.get('LLM_URL', 'http://localhost:8005')
LLM_MODEL = os.environ.get('LLM_MODEL', 'Qwen/Qwen3.5-9B')
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'base.en')
WHISPER_DEVICE = os.environ.get('WHISPER_DEVICE', 'cuda')
# Piper-only TTS (no CosyVoice speaker env)
PIPER_MODEL = os.environ.get('PIPER_MODEL', str(BASE / 'pretrained_models' / 'piper' / 'en_US-amy-medium.onnx'))
# Qwen3-TTS config
QWEN_TTS_MODEL = os.environ.get('QWEN_TTS_MODEL', 'pretrained_models/qwen3-tts-customvoice')
QWEN_TTS_SPEAKER = os.environ.get('QWEN_TTS_SPEAKER', 'Vivian')
QWEN_TTS_INSTRUCT = os.environ.get('QWEN_TTS_INSTRUCT', 'Speak clearly and naturally')

# In-memory conversation store: session_id -> list of {'role': 'user'|'assistant'|'system', 'content': '...'}
_conversations = {}
_MAX_TURNS = 20
_DEFAULT_SYSTEM_PROMPT = "You are Olly, a helpful and concise voice assistant. Keep your answers short and conversational — 1-3 sentences max. No bullet lists, no markdown, no internal reasoning, just natural speech. /no_think"

# Lazy-loaded globals
_cosy = None
_whisper = None
_tts_available = False
_whisper_load_attempted = False
_cosy_load_attempted = False

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

def load_cosy():
    global _cosy, _tts_available, _cosy_load_attempted
    if _cosy_load_attempted:
        return _cosy
    _cosy_load_attempted = True

    # --- Attempt 1: Qwen3-TTS ---
    try:
        import torch
        import soundfile as sf
        import numpy as np
        from qwen_tts import Qwen3TTSModel
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = Qwen3TTSModel.from_pretrained(
            QWEN_TTS_MODEL,
            device_map=device,
            dtype=dtype,
        )
        def inference_sft(text, **kwargs):
            import io, wave as wv
            wavs, sr = model.generate_custom_voice(
                text=text,
                language='English',
                speaker=QWEN_TTS_SPEAKER,
                instruct=QWEN_TTS_INSTRUCT,
            )
            audio = wavs[0]
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            # Normalize to int16
            if audio.dtype != np.int16:
                audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            buf = io.BytesIO()
            with wv.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio.tobytes())
            return buf.getvalue()
        _cosy = {'inference_sft': inference_sft}
        _tts_available = True
        print(f"Qwen3-TTS loaded: {QWEN_TTS_MODEL}, speaker={QWEN_TTS_SPEAKER}")
        return _cosy
    except Exception as e:
        print(f"Qwen3-TTS load failed ({e}), falling back to Piper")

    # --- Fallback: Piper ---
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
        print(f"Piper TTS load failed: {e}")
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
        print(f"WARNING: LLM server at {LLM_URL} is unreachable. /chat will return 502 until it's available. Set LLM_URL env var to override.")
    # load models in background threads so handlers don't block
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_whisper)
    await loop.run_in_executor(None, load_cosy)

@app.get('/health')
async def health():
    status = {'status': 'ok'}
    # report whisper/tts availability based on already-loaded globals
    status['whisper'] = 'ok' if _whisper is not None else 'missing'
    if _tts_available and _cosy:
        # detect which TTS is loaded
        try:
            from qwen_tts import Qwen3TTSModel
            status['tts'] = 'qwen3-tts'
        except ImportError:
            status['tts'] = 'piper'
    else:
        status['tts'] = 'stub'
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
    with wave.open(path, 'wb') as wf:
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
            # Piper-only: use inference_sft without speaker override
            wav_bytes = await loop.run_in_executor(None, lambda: _cosy['inference_sft'](text))
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
    suffix = Path(file.filename or '').suffix.lower() or '.bin'
    allowed_suffixes = {'.wav', '.mp3', '.m4a', '.webm', '.ogg', '.flac', '.mp4'}
    if suffix not in allowed_suffixes:
        suffix = '.webm'

    tmpdir = tempfile.gettempdir()
    fd, path = tempfile.mkstemp(suffix=suffix, dir=tmpdir)
    os.close(fd)
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            return JSONResponse({'error': 'Uploaded audio is empty.'}, status_code=400)

        with open(path,'wb') as f:
            f.write(audio_bytes)

        if model is None:
            return {'text': ''}

        # Use faster-whisper to transcribe in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        try:
            segments, info = await loop.run_in_executor(None, lambda: list(model.transcribe(path, beam_size=5)))
        except Exception as e:
            print('ERROR /transcribe: transcription failed')
            traceback.print_exc()
            return JSONResponse(
                {'error': f'Transcription failed: {type(e).__name__}: {e}'},
                status_code=422,
            )

        text = ''
        for segment in segments:
            text += segment.text
        return {'text': text}
    except Exception as e:
        print('ERROR /transcribe: request handling failed')
        traceback.print_exc()
        return JSONResponse({'error': f'Transcribe request failed: {type(e).__name__}: {e}'}, status_code=500)
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
    payload = {'messages': messages, 'model': LLM_MODEL, 'max_tokens': 300, 'stop': ['\nUser:', '\nuser:', '\nAssistant:', '\nassistant:']}
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(f"{LLM_URL}/chat", json=payload)
            r.raise_for_status()
            jr = r.json()
            print(f"[TIMING] LLM: {(_time.time()-_t0)*1000:.0f}ms")
            if isinstance(jr, dict) and 'choices' in jr:
                raw_reply = jr['choices'][0].get('message', {}).get('content') or jr['choices'][0].get('content') or ''
            elif isinstance(jr, dict) and 'message' in jr:
                msg = jr['message']
                raw_reply = msg.get('content', '') if isinstance(msg, dict) else str(msg)
            else:
                raw_reply = jr.get('text', '') if isinstance(jr, dict) else str(jr)
    except Exception as e:
        return JSONResponse({'error': f'LLM call failed: {e}'}, status_code=502)
    _t1 = _time.time()
    import re
    # Extract <think> block if present (Qwen3 formal thinking mode)
    think_match = re.search(r'<think>(.*?)</think>', raw_reply, flags=re.DOTALL)
    thinking_text = think_match.group(1).strip() if think_match else ''

    # Strip <think> tags from reply
    tts_reply = re.sub(r'<think>.*?</think>', '', raw_reply, flags=re.DOTALL).strip()

    # Strip inline reasoning patterns (Qwen3-0.6B outputs these without <think> tags)
    # Match first occurrence of reasoning leak and cut there
    reasoning_boundary = re.search(
        r'\s+(Okay,?\s+(?:the\s+)?(?:user|system)|I need to|I should|Let me|'
        r'The user (?:is asking|then says|says)|Since it\'s|Wait,|Actually,|Hmm,|'
        r'The system|\*\*Final Answer\*\*|Answer:|```)',
        tts_reply, flags=re.IGNORECASE
    )
    if reasoning_boundary:
        # Save what came before as the clean reply; save full as thinking
        if not thinking_text:
            thinking_text = tts_reply[reasoning_boundary.start():].strip()
        tts_reply = tts_reply[:reasoning_boundary.start()].strip()

    # Hard stop at role continuations for stop_marker in ['\nUser:', '\nuser:', '\nAssistant:', '\nassistant:', '\n\n\n']:
    for stop_marker in ['\nUser:', '\nuser:', '\nAssistant:', '\nassistant:', '\n\n\n']:
        if stop_marker in tts_reply:
            tts_reply = tts_reply[:tts_reply.index(stop_marker)].strip()

    # Remove repetition loops: if same sentence appears twice, cut at first repeat
    sentences = re.split(r'(?<=[.!?])\s+', tts_reply)
    seen_s: list[str] = []
    deduped: list[str] = []
    for s in sentences:
        norm = s.strip().lower()
        if not norm:
            continue
        if norm in seen_s:
            break
        seen_s.append(norm)
        deduped.append(s)
    tts_reply = ' '.join(deduped).strip()

    # Final safety: if tts_reply is empty (model only produced reasoning), use a fallback
    if not tts_reply:
        tts_reply = "I'm not sure how to answer that. Could you rephrase?"

    # Store clean reply in history
    reply = tts_reply
    # Append assistant reply to history
    _append_turn(session_id, 'assistant', reply)
    # Synthesize via TTS
    tmpdir = tempfile.gettempdir()
    fd, path = tempfile.mkstemp(suffix='.wav', dir=tmpdir)
    os.close(fd)
    try:
        if _tts_available and _cosy:
            loop = asyncio.get_event_loop()
            # Piper-only: use inference_sft without speaker override
            wav_bytes = await loop.run_in_executor(None, lambda: _cosy['inference_sft'](reply))
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
        safe_reply = tts_reply.replace('\n', ' ').replace('\r', '').encode('ascii', 'ignore').decode('ascii')[:500]
        safe_thinking = thinking_text.replace('\n', ' ').replace('\r', '').encode('ascii', 'ignore').decode('ascii')[:1000]
        headers = {
            'X-Assistant-Reply': safe_reply,
            'X-Thinking': safe_thinking,
            'X-Session-Id': session_id,
        }
        return StreamingResponse(iterfile(path), media_type='audio/wav', background=BackgroundTask(os.remove, path), headers=headers)
    except Exception as e:
        try:
            os.remove(path)
        except Exception:
            pass
        return JSONResponse({'error': str(e)}, status_code=500)

# --- Meeting endpoints ---

@app.post('/meeting/save')
async def save_meeting(body: dict):
    name = body.get('name') or f"meeting_{_dt.now().strftime('%Y%m%d_%H%M%S')}"
    transcript = body.get('transcript', '')
    duration = body.get('duration', '')
    ts = _dt.now().strftime('%Y-%m-%d %H:%M:%S')
    safe_name = ''.join(c if c.isalnum() or c in '-_ ' else '_' for c in name).strip()
    filename = f"{_dt.now().strftime('%Y%m%d_%H%M%S')}_{safe_name[:40]}.md"
    md = f"# {name}\n\nDate: {ts}\nDuration: {duration}\n\n## Transcript\n\n{transcript}\n"
    path = MEETINGS_DIR / filename
    path.write_text(md, encoding='utf-8')
    return JSONResponse({'status': 'saved', 'filename': filename})

@app.get('/meetings')
async def list_meetings():
    files = sorted(MEETINGS_DIR.glob('*.md'), reverse=True)
    return JSONResponse({'meetings': [{'filename': f.name, 'modified': f.stat().st_mtime} for f in files[:20]]})

@app.get('/meeting/{filename}')
async def get_meeting(filename: str):
    path = MEETINGS_DIR / filename
    if not path.exists() or path.suffix != '.md':
        return JSONResponse({'error': 'not found'}, status_code=404)
    return JSONResponse({'content': path.read_text(encoding='utf-8')})

# Serve the UI static files under /ui
ui_path = BASE / 'ui'
if ui_path.exists():
    app.mount('/ui', StaticFiles(directory=str(ui_path)), name='ui')

# Register tts_clone router if available (feature/tts-voice-clone)
try:
    from . import tts_clone
    app.include_router(tts_clone.router)
    print("Registered tts_clone router")
except Exception as e:
    print(f"tts_clone router not registered: {e}")
