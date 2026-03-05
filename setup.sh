#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn[standard]
# Optional installs for developer convenience (faster-whisper may be needed for
# transcription; CosyVoice is a separate/private component and must be installed
# manually into vendor/CosyVoice per project docs).
# Install faster-whisper (optional):
# pip install faster-whisper
# Note: CosyVoice is not installed automatically by this script. Add it under
# vendor/CosyVoice and follow its install instructions (private/stubbed for now).

# Instructions: add CosyVoice into vendor/CosyVoice and install its requirements
echo "Add vendor/CosyVoice as a submodule or copy. See SKILL.md for details."
