# Qwen Upgrade Spec — olly-voice-server

## Overview
Upgrade olly-voice-server to use Qwen3 family for both brain (Qwen3.5-9B) and high-quality TTS (Qwen3-TTS). Move TTS from local Piper -> Qwen3-TTS (primary: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice). Run the LLM (9B) on the Windows private-ai-server; keep olly-voice as lightweight FastAPI client that proxies /chat → private-ai-server, then synthesizes returned text.

## Step 1: Qwen3-TTS integration (olly-voice-server, Python side)

### 1.1 Compatibility check
- Dry-run pip output (from .venv Python 3.13) shows qwen-tts would pull modern transformers (4.57.3) and other deps; dry-run succeeded in resolving packages.
- qwen-tts is not currently installed in the venv ("not installed").
- Conclusion: qwen-tts appears installable in this environment (pip resolved deps) but may add large deps (transformers, huggingface_hub, gradio, accelerate). No explicit Python 3.13 incompatibility surfaced in dry-run. Risk: some transitive deps (transformers 4.57.3 etc.) may not be fully validated on 3.13; prepare fallback.

Workaround if incompatibility found at install/runtime:
- Run Qwen3-TTS inside a separate Python 3.10/3.11 container or small micro-service. Communicate via local HTTP (e.g., tcp:5006) or Unix socket. Keep current olly /tts endpoints unchanged; forward to qwen-tts service.

### 1.2 Installation
- Preferred (in-repo venv):
  .venv/bin/pip install -U qwen-tts

- Alternative (separate container):
  docker run --rm -it --gpus all -p 5006:5006 -v /path/to/models:/models python:3.11-slim bash
  pip install qwen-tts

### 1.3 Model download
- Use huggingface_hub to pre-download model to local storage (avoid on-the-fly download at first run):
  python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', cache_dir='pretrained_models/qwen3-tts')"
- Or rely on qwen-tts.from_pretrained to fetch on demand; prefer predownload for repeatable deploys.

### 1.4 app.py changes
- Replace or supplement load_cosy() to attempt loading Qwen3-TTS first, falling back to Piper. Provide a clear API surface compatible with existing `_cosy['inference_sft'](text, speaker=...)` usage.

Pseudocode diff (high level):
- New env vars: QWEN_TTS_MODEL, QWEN_TTS_SPEAKER, QWEN_TTS_INSTRUCT
- Add new loader function load_qwen_tts() and modify load_cosy() to prefer qwen-tts.

Example pseudocode to add into app.py (conceptual):

from qwen_tts import Qwen3TTSModel

def load_qwen_tts():
    global _cosy, _tts_available
    try:
        model_name = os.environ.get('QWEN_TTS_MODEL', 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice')
        speaker = os.environ.get('QWEN_TTS_SPEAKER', 'Vivian')
        model = Qwen3TTSModel.from_pretrained(model_name, device_map='cuda:0', dtype='bfloat16')
        def inference_sft(text, speaker_override=None, **kwargs):
            sp = speaker_override or speaker
            wavs, sr = model.generate_custom_voice(text=text, language='English', speaker=sp, instruct=os.environ.get('QWEN_TTS_INSTRUCT',''))
            # qwen-tts returns numpy/wav bytes or wav arrays; coerce to bytes
            return wavs[0]
        _cosy = {'inference_sft': inference_sft}
        _tts_available = True
        return _cosy
    except Exception as e:
        print('Qwen TTS load failed', e)
        return None

- Ensure the existing tts and chat endpoints call _cosy['inference_sft'] the same way (text, speaker=...). Keep the same return handling (bytes or path).

Streaming plan:
- qwen-tts supports stream=True in generate_custom_voice. For low-latency interactive audio, implement a streaming path: call model.generate_custom_voice(stream=True) and yield chunks to StreamingResponse. This may require an async wrapper.

### 1.5 health endpoint update
- Change health() to return 'tts': 'qwen3-tts' when Qwen model loaded, else 'piper' or 'stub'.

### 1.6 Environment variables
- QWEN_TTS_MODEL (default: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- QWEN_TTS_SPEAKER (default: Vivian)
- QWEN_TTS_INSTRUCT (default: '')
- Keep PIPER_MODEL for fallback.

## Step 2: Qwen3.5-9B on private-ai-server (Windows side)

### 2.1 VRAM assessment
- Local GPU: NVIDIA GeForce RTX 4090 Laptop GPU — total 16,376 MiB, free ~1,907 MiB (on current system). This is per nvidia-smi run from planner host. The private-ai-server Windows GPU(s) must be measured on that machine; planner must run nvidia-smi on Windows box to get accurate VRAM.
- Recommendation: Qwen3.5-9B raw fp16 requires ~20+ GB if loaded dense. Use quantization to fit into common GPUs.
- Quantization recommendation: use 4-bit quantization (BitsAndBytes or AWQ/GPTQ) to fit into 16GB GPUs. Prefer AWQ or GPTQ if supported; BitsAndBytes 4-bit (nf4+pg) via transformers+bitsandbytes is easiest.

### 2.2 Model loading strategy
- Preferred: vLLM if private-ai-server already uses it (faster, streaming, efficient). If private-ai-server uses transformers pipeline today, add support via:
  - transformers + accelerate + bitsandbytes (bnb 4bit) to load model with device_map='auto' and load_in_4bit=True
  - or use vLLM (recommended for throughput & streaming) with quantized GGUF model; convert model to GGUF if using llama-like tooling.
- For simplest path: use transformers + bitsandbytes 4-bit (GPTQ/AWQ optional) and set model_type appropriately.

### 2.3 private-ai-server changes
- Update model name and loading path in private-ai-server config (LLM_MODEL default -> Qwen/Qwen3.5-9B).
- Add optional quantization pipeline: if VRAM < threshold, load quantized weights (bnb 4-bit). Add flags to select quantization method.
- Install necessary deps on Windows GPU server: bitsandbytes, accelerate, transformers (matching versions), and any model-specific tokenizers.
- Test with small batch and measure latency.

### 2.4 Thinking mode handling
- Qwen3.5 may emit internal <think>...</think> blocks or chain-of-thought. Two mitigations:
  1. Use system prompt override: append '/no_think' instruction in the system prompt (current app already uses _DEFAULT_SYSTEM_PROMPT containing '/no_think'). Ensure private-ai-server respects system messages and doesn't enable chain-of-thought.
  2. Post-process server response: strip <think>...</think> blocks (app.py already strips via regex). Keep both layers.
- Also cap max_tokens (e.g., 200) and set stop tokens if server supports them.

### 2.5 API compatibility
- Keep private-ai-server /chat API compatible: olly-voice sends messages as currently implemented. private-ai-server should accept messages list + model + max_tokens and respond in the same schema. No olly-side changes needed if private-ai-server keeps same contract.

## Step 3: Docker Compose packaging

### 3.1 Services
- olly-voice service (FastAPI) — build from repo, mount pretrained_models or use model container for TTS if separated.
- private-ai-server service (Windows) — usually runs on separate host; if moved to Linux container, run as separate service with GPU access.

### 3.2 GPU passthrough
- Use nvidia-container-toolkit (docker run --gpus all) and set runtime to nvidia in compose:
  deploy:
    resources:
      reservations: { devices: [{ driver: nvidia, count: all, capabilities: [gpu] }] }
- For local dev with docker-compose v2: use deploy.resources or runtime: "nvidia" and environment NVIDIA_VISIBLE_DEVICES=all

### 3.3 Model volumes
- Mount pretrained_models/ into containers: e.g. - ./pretrained_models:/workspace/pretrained_models
- For private-ai-server host-mounted models, expose via NFS/SMB or copy to container image.

### 3.4 Networking
- Use an internal docker network; expose only UI port (e.g., 8000) to host. Keep private-ai-server internal or on separate host with restricted firewall.

## Risk Assessment
| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| qwen-tts Python 3.13 incompatible | Medium | Use Python 3.11 subprocess / separate container |
| 9B VRAM OOM | Medium | Use 4-bit quantization via BitsAndBytes; prefer vLLM or model sharding |
| Qwen3.5 thinking verbosity | Low | /no_think token + max_tokens cap + truncation |

## File Changes Summary
- olly_voice/app.py — add load_qwen_tts loader, prefer qwen-tts, streaming TTS path, health status change
- requirements.txt — add qwen-tts, transformers, huggingface_hub, bitsandbytes (optional), accelerate (optional)
- .env or Docker compose — add QWEN_TTS_MODEL, QWEN_TTS_SPEAKER, QWEN_TTS_INSTRUCT, LLM_MODEL override
- specs/qwen-upgrade.md — (this file)
- deployment/docker-compose.yml — add GPU runtime and model volume mounts (new file)

## Open Questions for Fabio
- Do you want TTS run inside the existing olly Python 3.13 venv or as a separate Python 3.11/3.10 service? (If separate, I'll plan an HTTP microservice and keep olly code minimal.)
- Which Qwen3-TTS speaker should be primary? I recommend "Vivian" (default premium warm female) — confirm or pick another.
- Where will the private-ai-server run (keep on Windows host or containerize on Linux)? I need the Windows GPU VRAM details to finalize quantization strategy.
- Do we want streaming TTS (lower latency) or simpler batch generation? Streaming is recommended for conversational UX.


## Key findings
- qwen-tts dry-run in .venv (Python 3.13) resolved packages and appears installable; not currently installed.
- Local planner machine GPU: NVIDIA GeForce RTX 4090 Laptop GPU — 16376 MiB total, 1907 MiB free at time of check. (Windows private-ai-server GPU must be measured separately.)
- Recommended TTS speaker: "Vivian" (default premium voice). Recommended LLM quantization: BitsAndBytes 4-bit (or AWQ/GPTQ if available) to fit Qwen3.5-9B on 16GB-class GPUs.

## Commit
Spec written and committed to specs/qwen-upgrade.md

