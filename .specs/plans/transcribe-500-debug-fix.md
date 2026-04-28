# Transcribe 500 Debug + Fix Plan

## Goal
Fix `POST /transcribe` returning `500` from the browser recording flow (`MediaRecorder` webm upload).

## Scope
- Backend: `olly_voice/app.py`
- Frontend error surfacing: `ui/index.html`

## Hypothesis
The `/transcribe` handler currently lets Whisper decode/transcribe exceptions bubble to FastAPI as a generic `500`. Browser uploads are `audio/webm`, and decode/runtime issues are not handled.

## Steps
- [x] Add guarded audio upload handling (empty file / malformed upload checks).
- [x] Add explicit transcription exception handling and log traceback server-side.
- [x] Return actionable JSON errors (not raw generic 500) for decode/transcribe failures.
- [x] Show backend-provided error message in UI when transcribe fails.
- [x] Validate endpoint behavior with a local syntax/import check.

## Acceptance Criteria
- `/transcribe` no longer fails with unhandled `500` for known decode/transcribe issues.
- Browser console shows actionable transcribe error message.
- Existing successful transcribe flow remains unchanged.
