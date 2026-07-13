"""Wire protocol for the /voice WebSocket.

Client → server:
  JSON text frames:
    {"type": "start", "format": "wav", "city": optional, "source_type": optional}
    {"type": "end_of_speech"}
  Binary frames: audio bytes (WAV; chunked frames are concatenated in order).

Server → client:
  JSON text frames:
    {"type": "partial_transcript", "text": str}
    {"type": "final_transcript", "text": str}
    {"type": "status", "node": str, "routes": [..]}          # pipeline progress
    {"type": "token", "text": str}                            # answer tokens
    {"type": "sentence", "text": str}                         # sentence sent to TTS
    {"type": "metrics", "asr_final_ms": f, "ttft_ms": f, "ttfa_ms": f, "total_ms": f}
    {"type": "result", "data": {AskResult dict}}
    {"type": "error", "message": str}
  Binary frames: one complete WAV per synthesized sentence, in playback order.

All latency metrics are measured from receipt of end_of_speech (t0).
"""

CLIENT_START = "start"
CLIENT_END_OF_SPEECH = "end_of_speech"

SERVER_PARTIAL_TRANSCRIPT = "partial_transcript"
SERVER_FINAL_TRANSCRIPT = "final_transcript"
SERVER_STATUS = "status"
SERVER_TOKEN = "token"
SERVER_SENTENCE = "sentence"
SERVER_METRICS = "metrics"
SERVER_RESULT = "result"
SERVER_ERROR = "error"
