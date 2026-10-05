"""
Pocket TTS provider wrapper (Kyutai pocket-tts).

Mirrors the qwen provider interface (load_model / synthesize_text / clone_voice)
so olly-voice-server can select "pocket" as an engine alongside the Qwen model.

Pocket TTS:
  - TTS-ONLY (no ASR). Small (~100M), CPU-optimized.
  - Voice cloning via an audio prompt -> safetensors state for fast reload.
  - sample_rate = 24000 Hz. Runs in-process; model + voice states cached.
  - Officially CPU; may .to("cuda") yourself though batch=1 usually no win.
"""
from pathlib import Path
import threading
import io
import wave
import os
import tempfile

_models = {}
_models_lock = threading.Lock()
# voice_state cache: {cache_key: voice_state}
_voice_cache = {}
_voice_lock = threading.Lock()

# Map friendly engine id -> pocket language config. "" = english default.
_LANG_MAP = {
    "english": "english",
    "italian": "italian",
    "italian_24l": "italian_24l",
    "french": "french",
    "german": "german",
    "spanish": "spanish",
    "portuguese": "portuguese",
    "dutch": "dutch",
}

# catalog voice set that pocket_tts accepts by name without cloning
_CATALOG_VOICES = {
    "cosette", "marius", "javert", "alba", "jean", "anna", "vera",
    "fantine", "charles", "paul", "eponine", "azelma", "george", "mary",
    "jane", "michael", "eve", "bill_boerst", "peter_yearsley",
    "stuart_bell", "caro_davy", "giovanni", "lola", "juergen", "rafael",
    "daan", "estelle",
}


def _to_wav_bytes(audio, sr):
    """Convert a 1D PCM tensor/array to WAV bytes (16-bit PCM mono)."""
    try:
        import numpy as np
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        if audio.dtype != np.int16:
            audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    except Exception:
        if isinstance(audio, bytes):
            return audio
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def load_model(model_path_or_id: str):
    """Load or return cached Pocket TTSModel instance.

    model_path_or_id is a *language config* name here (e.g. "italian_24l").
    Raises ImportError if pocket_tts is not installed.
    """
    global _models
    lang = _LANG_MAP.get(model_path_or_id, model_path_or_id or "english")
    if lang in _models:
        return _models[lang]
    try:
        from pocket_tts import TTSModel
    except Exception as e:
        raise ImportError(f"pocket_tts package not available: {e}")
    with _models_lock:
        if lang in _models:
            return _models[lang]
        model = TTSModel.load_model(language=lang)
        model._pocket_lang = lang  # annotate for synthesize_text
        _models[lang] = model
        return model


def _model_language(model) -> str:
    return getattr(model, "_pocket_lang", "english")


def _get_voice_state(model, sample_wav_path: str):
    """Extract (and cache) a voice-cloning state for a sample wav.

    Falls back to catalog 'alba' if cloning unavailable so TTS still works.
    """
    key = f"{_model_language(model)}::{sample_wav_path or 'alba'}"
    with _voice_lock:
        if key in _voice_cache:
            return _voice_cache[key]
    state = model.get_state_for_audio_prompt(sample_wav_path or "alba")
    with _voice_lock:
        _voice_cache[key] = state
    return state


def synthesize_text(model, text: str, speaker: str = None, instruct: str = None, language: str = None):
    """Synthesize text to WAV bytes using Pocket TTS.

    speaker = catalog voice name (default 'alba') OR a path to a wav sample
    used for cloning. language = pocket language config (default model's).
    """
    # Prefer explicit language; load that model variant if different.
    lang = language or _model_language(model)
    if lang in _LANG_MAP and lang != _model_language(model):
        model = load_model(lang)

    voice_ref = speaker or "alba"
    if voice_ref in _CATALOG_VOICES:
        state = model.get_state_for_audio_prompt(voice_ref)
    else:
        state = _get_voice_state(model, voice_ref)
    audio = model.generate_audio(state, text)
    return _to_wav_bytes(audio, model.sample_rate)


def clone_voice(model, sample_wav_path: str, text: str, ref_text: str = None, speaker: str = None, instruct: str = None, language: str = None, x_vector_only_mode: bool = True):
    """Voice clone from a sample WAV via Pocket TTS, return WAV bytes."""
    if not sample_wav_path or not Path(sample_wav_path).exists():
        if sample_wav_path:
            print(f"[pocket] clone sample not found: {sample_wav_path}; using default voice", flush=True)
        return synthesize_text(model, text, speaker="alba", language=language)
    state = _get_voice_state(model, sample_wav_path)
    audio = model.generate_audio(state, text)
    return _to_wav_bytes(audio, model.sample_rate)


def export_fabio_voice(sample_wav_path: str, dest_dir: str = None, language: str = "english"):
    """Convenience: export a voice-state safetensors for fast reload.

    Pre-builds Fabio's voice states (fabio-en / fabio-it) for near-instant load.
    """
    from pocket_tts import export_model_state
    model = load_model(language)
    state = model.get_state_for_audio_prompt(sample_wav_path)
    dest_dir = dest_dir or os.path.join(tempfile.gettempdir(), "pocket-voices")
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, f"voice-{language}.safetensors")
    export_model_state(state, out)
    return out


def get_supported_speakers(model):
    return sorted(_CATALOG_VOICES)


def get_supported_languages(model):
    return list(_LANG_MAP.keys())
