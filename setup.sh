#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn[standard]
# Instructions: add CosyVoice into vendor/CosyVoice and install its requirements
echo "Add vendor/CosyVoice as a submodule or copy. See SKILL.md for details."
