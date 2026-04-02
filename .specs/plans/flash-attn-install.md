# Plan: Install flash-attn for faster Qwen TTS inference

Path: repositories/olly-voice-server/.specs/plans/flash-attn-install.md
Owner: Olly
Branch: feature/install-flash-attn
Created: 2026-03-29

Goal
----
Install and verify flash-attn in the olly-voice-server virtual environment to reduce GPU memory pressure and speed up Qwen-TTS inference (especially for the 1.7B model). Provide a reproducible, reversible procedure with tests and a short benchmark (p50/p95 latencies and GPU memory sample).

Summary of steps
----------------
1. Inspect current environment (CUDA, PyTorch) and record baseline performance.
2. Try installing the prebuilt wheel via pip inside the olly-voice-server venv. If that fails, build from source with the correct CUDA/arch flags.  
3. Verify installation and run short before/after inference benchmarks.  
4. Add an automated smoke-test and a /todo entry (done).  

Prerequisites & notes
---------------------
- Requires a CUDA-enabled GPU and a matching PyTorch/CUDA compatible wheel for flash-attn. We have an RTX 4090 (16GB) in this machine.  
- Confirm PyTorch: python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"  
- Potential version matrix issues: flash-attn binary wheels are distributed for specific (PyTorch, CUDA) versions. Building from source requires a working compiler toolchain (gcc, nvcc) and matching CUDA toolkit headers.  
- flash-attn is a performance optimization only — not functionally required. Keep a rollback plan.

Estimated time
--------------
- Quick pip install (if wheel matches): ~5–15 minutes including verification.  
- Build from source (if pip wheel not available): 20–60 minutes depending on machine and toolchain.  

Detailed steps
--------------
1) Baseline measurements (record these before changes)
   - Record GPU/PyTorch versions:
     - nvidia-smi
     - python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   - Run a short Qwen-TTS inference script (use trimmed 3s sample and a 3–8 word target) on 1.7B with device_map='cuda:0', dtype=torch.bfloat16 and measure wall-clock. Save outputs to ~/.openclaw/.benchmarks/flash-attn/baseline.txt
   - Record GPU memory usage during run (nvidia-smi --query-compute-apps=pid,used_memory --format=csv) or use torch.cuda.max_memory_allocated() in-script.

2) Attempt pip wheel install (recommended first)
   - Activate venv: source repositories/olly-voice-server/.venv/bin/activate
   - Try: pip install --upgrade pip
   - Try installing compatible wheel:
     - pip install flash-attn
   - If pip succeeds, verify import: python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"

3) If pip wheel fails → Build from source
   - Ensure toolchain available: gcc, g++, make, cmake, nvcc (CUDA toolkit). On Ubuntu/Debian: sudo apt install build-essential cmake git
   - Fetch flash-attn: git clone https://github.com/hazyResearch/flash-attention.git && cd flash-attention
   - Follow README build steps inside the olly-voice-server venv. Example: python setup.py install or pip install .
   - You may need to set TORCH_CUDA_ARCH_LIST or specific CUDA home env vars:
     - export TORCH_CUDA_ARCH_LIST="8.6"  (adjust to your GPU arch compute capability; RTX 4090 uses 8.9/8.6 depending on model)
     - export CUDA_HOME=/usr/local/cuda
   - Build (example): python setup.py build_ext --inplace && pip install -e .

4) Verify & benchmark
   - Run the same short inference script as in baseline and compare wall-clock times (median of 5 runs) and GPU memory usage. Save to ~/.openclaw/.benchmarks/flash-attn/after_install.txt
   - Expected: reduced inference latency (1.5x–3x typical) and lower peak GPU memory.

5) Rollback (if regressions or issues)
   - pip uninstall flash-attn
   - Re-run baseline test to confirm previous behavior restored.

6) Automation & tests
   - Add a small smoke-test in repositories/olly-voice-server/tests/test_flash_attn_smoke.py that:
     - imports flash_attn when available, otherwise skips
     - runs a tiny inference on a short sample and asserts audio output exists
   - Add benchmark scripts to ~/.openclaw/.benchmarks/flash-attn/ and document how to run them.

7) Post-install changes (optional)
   - If performance is improved, update README and .specs with results and set recommended dtype/device defaults (e.g., use bfloat16 on CUDA when available).
   - Consider enabling flash-attn usage detection in olly-voice-server startup (print if available).

Security & safety
-----------------
- Building from source requires developer toolchain — be cautious on production machines.  
- Do not store or expose any HF tokens or secrets while performing installs.

Files to add / branch policy
---------------------------
- Create branch: feature/install-flash-attn
- Files created/updated (on branch):
  - repositories/olly-voice-server/.specs/plans/flash-attn-install.md  (this file)
  - repositories/olly-voice-server/.specs/benchmarks/flash-attn/ (benchmark outputs)
  - repositories/olly-voice-server/tests/test_flash_attn_smoke.py (optional)

Verification checklist (when done)
----------------------------------
- [ ] pip import flash_attn succeeds inside venv
- [ ] Benchmark shows latency improvement or lower peak VRAM
- [ ] Smoke-test passes (if added)
- [ ] Plan file updated with measured numbers and notes

References
----------
- flash-attn repo: https://github.com/flash-attention/flash-attention
- common build notes: https://github.com/flash-attention/flash-attention#installation


Recorded plan path: repositories/olly-voice-server/.specs/plans/flash-attn-install.md

