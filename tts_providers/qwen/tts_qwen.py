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
    """Synthesize text to wav bytes using a suitable generate API on the model.
    Returns bytes of a WAV (16-bit PCM) file.
    """
    # Try preferred APIs in order, handling models that implement different subsets
    # 1) generate_custom_voice
    # 2) generate_voice_design
    # 3) generate (fallback)
    tried = []
    def _to_wav_bytes(audio, sr):
        try:
            import numpy as np
            if hasattr(audio, 'numpy'):
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

    # Helper to attempt a call and return wav bytes or raise
    if hasattr(model, 'generate_custom_voice'):
        try:
            wavs, sr = model.generate_custom_voice(text=text, language=language or 'English', speaker=speaker, instruct=instruct)
            return _to_wav_bytes(wavs[0], sr)
        except ValueError as e:
            tried.append(('generate_custom_voice', str(e)))
        except TypeError:
            tried.append(('generate_custom_voice', 'signature mismatch'))

    if hasattr(model, 'generate_voice_design'):
        try:
            wavs, sr = model.generate_voice_design(text=text, language=language or 'English', speaker=speaker, instruct=instruct)
            return _to_wav_bytes(wavs[0], sr)
        except Exception as e:
            tried.append(('generate_voice_design', str(e)))

    if hasattr(model, 'generate'):
        try:
            wavs, sr = model.generate(text=text, language=language or 'English', speaker=speaker, instruct=instruct)
            return _to_wav_bytes(wavs[0], sr)
        except Exception as e:
            tried.append(('generate', str(e)))

    # If we get here, nothing worked
    raise RuntimeError(f"No supported generate method succeeded. Attempts: {tried}")


def clone_voice(model, sample_wav_path: str, text: str, speaker: str = None, instruct: str = None, language: str = None, x_vector_only_mode: bool = True):
    """Perform voice cloning given a sample WAV path and text. Returns WAV bytes.
    Tries model.generate_voice_clone or model.generate_custom_voice with common sample args.
    x_vector_only_mode: if True, use speaker embedding only (no ICL); if False, use ICL mode.
    """
    def _to_wav_bytes(audio, sr):
        try:
            import numpy as np
            if hasattr(audio, 'numpy'):
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

    if hasattr(model, 'generate_voice_clone'):
        try:
            # primary expected arg name: ref_audio
            wavs, sr = model.generate_voice_clone(text=text, ref_audio=sample_wav_path, language=language or 'English', speaker=speaker, instruct=instruct, x_vector_only_mode=x_vector_only_mode)
            return _to_wav_bytes(wavs[0], sr)
        except TypeError:
            # some models use different param names (try a few common variants)
            for alt in ('sample', 'reference_wav', 'ref_audio', 'ref_wav'):
                try:
                    wavs, sr = model.generate_voice_clone(text=text, **{alt: sample_wav_path}, language=language or 'English', speaker=speaker, instruct=instruct, x_vector_only_mode=x_vector_only_mode)
                    return _to_wav_bytes(wavs[0], sr)
                except TypeError:
                    continue
                except Exception:
                    pass
        except Exception as e:
            raise

    # Try generate_custom_voice with common sample parameter names (most custom_voice models do not support cloning)
    if hasattr(model, 'generate_custom_voice'):
        for arg in ('audio_sample', 'reference_wav', 'sample_audio', 'audio_reference', 'audio_prompt', 'sample'):
            try:
                wavs, sr = model.generate_custom_voice(text=text, language=language or 'English', speaker=speaker, instruct=instruct, **{arg: sample_wav_path})
                return _to_wav_bytes(wavs[0], sr)
            except TypeError:
                continue
            except Exception as e:
                raise

    raise RuntimeError('Model does not support voice cloning via known APIs')


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
