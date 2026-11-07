import io
import json
import os
import unittest
import urllib.parse
import urllib.request
import uuid
import wave


@unittest.skipUnless(os.environ.get("OLLY_LIVE_SMOKE") == "1", "Requires a running local Olly service")
class LiveVoiceSmokeTest(unittest.TestCase):
    base_url = "http://127.0.0.1:8766"

    def request(self, path, *, data=None, headers=None, method=None):
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers or {}, method=method
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.read()

    def assert_audible_wav(self, audio):
        with wave.open(io.BytesIO(audio), "rb") as wav:
            self.assertGreater(wav.getnframes(), 0)
            self.assertTrue(any(wav.readframes(wav.getnframes())))

    def test_synthetic_speech_transcription_and_grounded_reply(self):
        form = urllib.parse.urlencode({"text": "Please explain what a loop is."}).encode()
        speech = self.request(
            "/tts", data=form, headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        self.assert_audible_wav(speech)

        boundary = "olly-smoke-test-boundary"
        upload = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="speech.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode() + speech + f"\r\n--{boundary}--\r\n".encode()
        transcript = json.loads(self.request(
            "/transcribe", data=upload, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        ))
        self.assertTrue(transcript["text"].strip())

        session_id = f"smoke-{uuid.uuid4()}"
        try:
            payload = json.dumps({
                "text": transcript["text"],
                "session_id": session_id,
                "system_prompt": "Explain only this synthetic lesson: a loop repeats a set of instructions.",
            }).encode()
            reply = self.request(
                "/chat", data=payload, headers={"Content-Type": "application/json"}
            )
            self.assert_audible_wav(reply)
        finally:
            self.request(f"/history/{session_id}", method="DELETE")


if __name__ == "__main__":
    unittest.main()
