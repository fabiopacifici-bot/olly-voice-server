"""
Qwen TTS provider wrapper.
Provides load_model, synthesize_text, and clone_voice helpers.
This module is optional and will fail gracefully if qwen_tts is not installed.
"""
from pathlib import Path
import threading
import os
import io
import wave

_models = {}
_models_lock = threading.Lock()


def load_model(model_path_or_id: str):
    """Load or return cached Qwen3TTSModel instance.
    model_path_or_id may be a local path or a Hugging Face repo id.
    Returns model object or raises ImportError if qwen_tts not available.
    """
    global _models
    if model_path_or_id in _models:
        return _models[model_path_or_id]
    try:
        from qwen_tts import Qwen3TTSModel
    except Exception as e:
        raise ImportError(f"qwen_tts package not available: {e}")
    # Simple caching
    with _models_lock:
        if model_path_or_id in _models:
            return _models[model_path_or_id]
        # Allow local path or hf id
        model = Qwen3TTSModel.from_pretrained(model_path_or_id)
        _models[model_path_or_id] = model
        return model


def synthesize_text(model, text: str, speaker: str = None, instruct: str = None, language: str = None):
    """Synthesize text to wav bytes using model.generate_custom_voice or model.generate.
    Returns bytes of a WAV (16-bit PCM) file.
    """
    # Prefer model.generate_custom_voice if available (as used in app.py)
    gen = None
    if hasattr(model, 'generate_custom_voice'):
        gen = model.generate_custom_voice
        wavs, sr = gen(text=text, language=language or 'English', speaker=speaker, instruct=instruct)
    elif hasattr(model, 'generate'):
        wavs, sr = model.generate(text=text, speaker=speaker, instruct=instruct)
    else:
        raise RuntimeError('Model does not expose known generate API')
    audio = wavs[0]
    # Normalise numpy -> int16 if needed
    try:
        import numpy as np
        if hasattr(audio,'numpy'):
            audio = audio.numpy()
        if audio.dtype != np.int16:
            audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    except Exception:
        # If numpy not available, assume bytes
        if isinstance(audio, bytes):
            return audio
    # Write to wav buffer
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def clone_voice(model, sample_wav_path: str, text: str, speaker: str = None, instruct: str = None, language: str = None):
    """Perform voice cloning given a sample WAV path and text. Returns WAV bytes.
    This uses the same generate_custom_voice API but passes additional keyword if supported.
    """
    # If model supports a clone API, call it; otherwise try generate_custom_voice with sample
    if hasattr(model, 'generate_custom_voice'):
        # Many qwen_tts variants accept an audio_base or similar; try common arg names
        kwargs = {}
        # try known parameter names gracefully
        for arg in ('audio_sample', 'reference_wav', 'sample_audio', 'audio_reference', 'audio_prompt'):
            try:
                wavs, sr = model.generate_custom_voice(text=text, language=language or 'English', speaker=speaker, instruct=instruct, **{arg: sample_wav_path})
                audio = wavs[0]
                try:
                    import numpy as np
                    if hasattr(audio,'numpy'):
                        audio = audio.numpy()
                    if audio.dtype != np.int16:
                        audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                except Exception:
                    if isinstance(audio, bytes):
                        return audio
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(audio.tobytes())
                return buf.getvalue()
            except TypeError:
                # parameter name not accepted, try next
                continue
            except Exception as e:
                # generation failed; raise for caller to handle
                raise
    # If we reach here, cloning not supported via generate_custom_voice signature
    raise RuntimeError('Model does not support cloning via known APIs')


def get_supported_speakers(model):
    try:
        return model.get_supported_speakers()
    except Exception:
        return []


def get_supported_languages(model):
    try:
        return model.get_supported_languages()
    except Exception:
        return []
