# Olly Voice Server — API Reference

Local AI voice assistant running on `http://127.0.0.1:8766`.

Provides: voice cloning (Qwen3-TTS), speech transcription (Whisper), text-to-speech, chat with LLM, and meeting recording/reconciliation.

---

## Base URL

```
http://127.0.0.1:8766
```

---

## Endpoints

### `GET /health`

Server liveness check.

**Response:**
```json
{
  "status": "ok",
  "whisper": "ok",
  "tts": "stub",
  "llm": "unreachable"
}
```

Fields reflect live state: `"ok"` | `"stub"` | `"unreachable"` | `"error"`.

---

### `POST /tts/clone` ⭐ Voice Cloning

Clone a voice from a WAV sample and synthesise text. Returns a WAV audio stream.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | ✅ | Text to synthesise |
| `model` | string | ✅ | `"0.6"` (fast, lighter) or `"1.7"` (higher quality) |
| `sample` | file (WAV) | ✅* | Reference voice WAV for cloning. Required for cloning — without it falls back to synthesis (may fail on base models) |
| `x_vector_only` | bool | ❌ | Default `true`. Use x-vector mode (recommended) |
| `ref_text` | string | ❌ | Transcript of the sample WAV. If omitted, auto-transcribed locally via Whisper |
| `speaker` | string | ❌ | Optional speaker ID |
| `instruct` | string | ❌ | Optional instruction for style/emotion |
| `language` | string | ❌ | Optional language code (e.g. `"en"`, `"it"`) |

**Response:** `audio/wav` stream (16-bit PCM, mono, 24000 Hz)

**Example:**
```bash
curl -X POST http://127.0.0.1:8766/tts/clone \
  -F "text=Hello, this is a test of the voice clone." \
  -F "model=0.6" \
  -F "x_vector_only=true" \
  -F "sample=@/path/to/voice-sample.wav" \
  -o output.wav
```

**Voice samples location:** `~/.openclaw/media/voice-samples/`

| Sample | Language | Notes |
|---|---|---|
| `fabio-en-phonetic.wav` | English | Use for English narration |
| `fabio-ita-phonetic.wav` | Italian | Use for Italian narration |

**Error responses:**
```json
{ "error": "qwen provider not available on this server" }   // 503 — model not loaded
{ "error": "model not configured on server" }               // 400 — bad model value
{ "error": "No supported generate method succeeded..." }    // 500 — no sample provided to base model
```

---

### `POST /tts`

Basic TTS without voice cloning. Uses the configured Piper/CosyVoice model.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | ✅ | Text to synthesise |

**Response:** `audio/wav` stream

> Note: Returns a silent WAV if no TTS model is loaded.

---

### `POST /transcribe`

Transcribe an audio file to text using Whisper.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | audio file | ✅ | Audio file to transcribe (WAV, MP3, etc.) |

**Response:**
```json
{ "text": "transcribed text here" }
```

---

### `POST /chat`

Chat with the local LLM (Qwen3). Maintains conversation history per session.

**Content-Type:** `application/json`

**Request:**
```json
{
  "text": "What is machine learning?",
  "session_id": "my-session",
  "system_prompt": "You are a helpful assistant."
}
```

| Field | Required | Default |
|---|---|---|
| `text` | ✅ | — |
| `session_id` | ❌ | `"default"` |
| `system_prompt` | ❌ | Built-in default |

**Response:**
```json
{
  "reply": "Machine learning is...",
  "thinking": "",
  "tts_audio_url": "/history/my-session/last-tts.wav"
}
```

---

### `GET /history/{session_id}`

Get conversation history for a session.

**Response:**
```json
[
  { "role": "user", "content": "Hello" },
  { "role": "assistant", "content": "Hi there!" }
]
```

### `DELETE /history/{session_id}`

Clear conversation history for a session.

---

### `POST /meeting/save`

Save a meeting transcript.

### `POST /meeting/reconcile`

Auto-reconcile a meeting recording — extracts action items, summaries, decisions.

### `GET /meetings`

List saved meetings.

### `GET /meeting/{filename}`

Get a specific meeting file.

### `POST /meeting/autosave`

Auto-save meeting in progress.

### `GET /meeting/autosaves`

List auto-saved meetings.

### `POST /meeting/autosave/recover`

Recover from an auto-saved state.

### `DELETE /meeting/autosave/{filename}`

Delete an auto-save file.

### `POST /meeting/stream-chunk`

Stream a meeting audio chunk for real-time transcription.

### `GET /chats`

List all chat sessions.

---

## Model configuration

Models are loaded **lazily** on first request (not at startup).

| Model | Env var | Default path |
|---|---|---|
| Qwen3-TTS 0.6B | `QWEN_TTS_0_6_MODEL` | `/mnt/e/models/qwen3-tts/0.6B/...` |
| Qwen3-TTS 1.7B | `QWEN_TTS_1_7_MODEL` | `/mnt/e/models/qwen3-tts/1.7B/...` |
| Whisper | `WHISPER_MODEL` | Auto-detected |

Set via env vars before starting the server or in the systemd/nohup launch command.

---

## Starting the server

```bash
cd repositories/olly-voice-server
export QWEN_TTS_0_6_MODEL=/mnt/e/models/qwen3-tts/0.6B/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc
export QWEN_TTS_1_7_MODEL=/mnt/e/models/qwen3-tts/1.7B/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3
nohup .venv/bin/uvicorn olly_voice.app:app --host 0.0.0.0 --port 8766 > /tmp/olly_voice_server.log 2>&1 &
```

Or via the unified dashboard service control API:
```bash
curl -X POST http://localhost:8000/api/services/8766/start
```

---

## Integration with fabiopacifici.com platform

The platform uses `POST /tts/clone` to generate slide narration audio with Fabio's cloned voice.

- **Settings key:** `olly_voice_server_url` (default: `http://127.0.0.1:8766`)
- **Service class:** `App\Services\OllyVoiceServerService`
- **Voice sample:** `storage/app/private/voice-samples/voice-sample.wav`
- **Trigger:** per-slide, only when provider select is set to `voice-clone` in the slide editor
- **Global default** (OpenAI TTS) is unaffected

See: `fabiopacifici/.specs/plans/private-ai-server-voice-clone.md`
