from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.background import BackgroundTask
from pathlib import Path
import tempfile
import os
import shutil
import asyncio

router = APIRouter()

# Voice samples directory — overridable via VOICE_SAMPLES_DIR env var.
# Defaults to <repo>/media/voice-samples/ so a fresh install works out of the box.
_BASE = Path(__file__).resolve().parent.parent
VOICE_SAMPLES_DIR = Path(os.environ.get('VOICE_SAMPLES_DIR', str(_BASE / 'media' / 'voice-samples')))
VOICE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


@router.get('/tts/voices')
async def list_voices():
    """Return the list of available WAV voice samples that can be used for cloning."""
    samples = sorted(p.name for p in VOICE_SAMPLES_DIR.glob('*.wav'))
    return JSONResponse({'voices': samples})


@router.post('/tts/clone/voices')
async def upload_voice(file: UploadFile = File(...)):
    """Upload a WAV voice sample to the voice-samples directory."""
    filename = Path(file.filename or '').name
    if not filename:
        return JSONResponse({'error': 'filename is required'}, status_code=400)
    if Path(filename).suffix.lower() != '.wav':
        return JSONResponse({'error': 'only .wav files are accepted'}, status_code=415)
    # Prevent path traversal
    dest = VOICE_SAMPLES_DIR / filename
    if dest.parent.resolve() != VOICE_SAMPLES_DIR.resolve():
        return JSONResponse({'error': 'invalid filename'}, status_code=400)

    content = await file.read()
    if not content:
        return JSONResponse({'error': 'uploaded file is empty'}, status_code=400)

    with open(dest, 'wb') as f:
        f.write(content)

    return JSONResponse({'status': 'uploaded', 'voice': filename}, status_code=201)

# lazy provider registry: engine name -> module
_providers = {}
_provider_attempted = set()


def _ensure_provider(engine: str = "qwen"):
    """Return the provider module for an engine ("qwen" | "pocket").

    Caches imports. Returns None (and logs) if the engine failed to import.
    """
    global _providers, _provider_attempted
    if engine in _providers:
        return _providers[engine]
    if engine in _provider_attempted:
        return None
    _provider_attempted.add(engine)
    try:
        if engine == "pocket":
            from tts_providers.pocket import tts_pocket as provider
        else:
            # Use absolute import to work when running as an app module
            from tts_providers.qwen import tts_qwen as provider
        _providers[engine] = provider
        return provider
    except Exception as e:
        print(f"Failed to import {engine} provider: {e}")
        return None


def _split_engine_model(model: str):
    """Parse "<engine>:<model>" -> (engine, model).

    e.g. "pocket:italian_24l" -> ("pocket", "italian_24l"), "0.6" -> ("qwen", "0.6").
    """
    if not model:
        return "qwen", model
    if model.startswith("pocket"):
        rest = model.split(":", 1)[1] if ":" in model else "english"
        return "pocket", (rest or "english")
    return "qwen", model


@router.post('/tts/clone')
async def tts_clone(
    text: str = Form(...),
    model: str = Form('0.6'),
    speaker: str = Form(None),
    instruct: str = Form(None),
    sample: UploadFile = File(None),
    x_vector_only: bool = Form(True),
    language: str = Form(None),
    ref_text: str = Form(None),
):
    """Clone a voice from sample (optional) and synthesize text.

    model: '0.6'/1.7' (qwen) or 'pocket[:<lang>]' (pocket engine), or a model path/id.
    sample: optional wav file (small) for cloning.
    ref_text: optional transcript of the sample to improve prosody. If not provided and a sample is present, the server will transcribe the sample locally.
    Returns WAV stream.
    """
    engine, model_core = _split_engine_model(model)
    prov = _ensure_provider(engine)
    if prov is None:
        return JSONResponse({'error': f'{engine} provider not available on this server'}, status_code=503)

    if engine == "pocket":
        # Pocket engine: model_core IS the language config (e.g. "italian_24l")
        model_id = model_core
        if not model_id:
            return JSONResponse({'error': 'model not configured on server'}, status_code=400)
    else:
        # Map short names to env paths
        base = Path(__file__).resolve().parent.parent
        env_map = {
            '0.6': os.environ.get('QWEN_TTS_0_6_MODEL', os.environ.get('QWEN_TTS_MODEL')),
            '1.7': os.environ.get('QWEN_TTS_1_7_MODEL', os.environ.get('QWEN_TTS_MODEL')),
        }
        model_id = env_map.get(model_core, model_core)
        if not model_id:
            return JSONResponse({'error': 'model not configured on server'}, status_code=400)

    # Load model (may raise ImportError which we'll surface)
    try:
        mdl = prov.load_model(model_id)
    except Exception as e:
        return JSONResponse({'error': f'failed to load model: {e}'}, status_code=500)

    tmpdir = tempfile.gettempdir()
    fd, out_path = tempfile.mkstemp(suffix='.wav', dir=tmpdir)
    os.close(fd)

    try:
        if sample is not None:
            # save sample to temp file
            fd2, sample_path = tempfile.mkstemp(suffix='.wav', dir=tmpdir)
            os.close(fd2)
            with open(sample_path, 'wb') as f:
                f.write(await sample.read())

            # If ref_text not provided, transcribe locally using the server's whisper loader
            if not ref_text:
                try:
                    # local import to avoid cycle at module import time
                    from .app import load_whisper
                    whisper_model = load_whisper()
                    if whisper_model is not None:
                        loop = asyncio.get_event_loop()
                        segments = await loop.run_in_executor(None, lambda: list(whisper_model.transcribe(sample_path, beam_size=5)))
                        # join segment texts
                        ref_text = ''.join([s.text for s in segments])
                    else:
                        ref_text = ''
                except Exception as e:
                    # transcription failure should not block cloning; log and continue with empty ref_text
                    print(f"local transcription failed: {e}")
                    ref_text = ''

            try:
                wav_bytes = prov.clone_voice(mdl, sample_path, text=text, ref_text=ref_text, speaker=speaker, instruct=instruct, language=language, x_vector_only_mode=x_vector_only)
            finally:
                try:
                    os.remove(sample_path)
                except Exception:
                    pass
        else:
            wav_bytes = prov.synthesize_text(mdl, text=text, speaker=speaker, instruct=instruct, language=language)

        with open(out_path, 'wb') as f:
            f.write(wav_bytes)

        def iterfile(p):
            with open(p,'rb') as fh:
                while True:
                    chunk = fh.read(8192)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(iterfile(out_path), media_type='audio/wav', background=BackgroundTask(os.remove, out_path))
    except Exception as e:
        try:
            os.remove(out_path)
        except Exception:
            pass
        return JSONResponse({'error': str(e)}, status_code=500)
