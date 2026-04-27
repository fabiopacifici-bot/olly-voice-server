from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.background import BackgroundTask
from pathlib import Path
import tempfile
import os
import shutil
import asyncio

router = APIRouter()

# lazy import of provider
_provider = None
_provider_load_attempted = False


def _ensure_provider():
    global _provider, _provider_load_attempted
    if _provider is not None:
        return _provider
    if _provider_load_attempted:
        return None
    _provider_load_attempted = True
    try:
        # Use absolute import to work when running as an app module
        from tts_providers.qwen import tts_qwen as provider
        _provider = provider
        return _provider
    except Exception as e:
        print(f"Failed to import qwen provider: {e}")
        _provider = None
        return None


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

    model: '0.6' or '1.7' or a model path/id. If '0.6' or '1.7', the server will map to configured env vars.
    sample: optional wav file (small) for cloning.
    ref_text: optional transcript of the sample to improve prosody. If not provided and a sample is present, the server will transcribe the sample locally.
    Returns WAV stream.
    """
    prov = _ensure_provider()
    if prov is None:
        return JSONResponse({'error': 'qwen provider not available on this server'}, status_code=503)

    # Map short names to env paths
    base = Path(__file__).resolve().parent.parent
    env_map = {
        '0.6': os.environ.get('QWEN_TTS_0_6_MODEL', os.environ.get('QWEN_TTS_MODEL')),
        '1.7': os.environ.get('QWEN_TTS_1_7_MODEL', os.environ.get('QWEN_TTS_MODEL')),
    }
    model_id = env_map.get(model, model)
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
