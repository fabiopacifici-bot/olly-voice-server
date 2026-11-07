import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from olly_voice import app as voice_app


class VoiceModelStartupTest(unittest.IsolatedAsyncioTestCase):
    async def test_warms_whisper_and_tts(self):
        with patch.object(voice_app, "load_whisper") as whisper, patch.object(
            voice_app, "load_cosy"
        ) as tts:
            await voice_app.warm_voice_models()

        whisper.assert_called_once_with()
        tts.assert_called_once_with()

    async def test_startup_schedules_model_warmup(self):
        initialized = asyncio.Event()

        async def warm_models():
            initialized.set()

        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.get = AsyncMock()

        with patch.object(voice_app.httpx, "AsyncClient", return_value=client), patch.object(
            voice_app, "warm_voice_models", side_effect=warm_models
        ):
            await voice_app.startup_event()
            await asyncio.wait_for(initialized.wait(), timeout=1)

        client.get.assert_awaited_once()

    async def test_health_reports_the_loaded_tts_engine(self):
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.get = AsyncMock()

        with patch.object(voice_app.httpx, "AsyncClient", return_value=client), patch.object(
            voice_app, "_tts_available", True
        ), patch.object(voice_app, "_cosy", {"engine": "piper"}):
            response = await voice_app.health()

        self.assertEqual(json.loads(response.body)["tts"], "piper")


if __name__ == "__main__":
    unittest.main()
