from __future__ import annotations

import hashlib
import hmac
import json

import numpy as np


def window_digest(window: np.ndarray, event_id: str) -> str:
    values = np.ascontiguousarray(np.asarray(window, dtype=np.float32))
    header = json.dumps(
        {"event_id": event_id, "shape": list(values.shape), "dtype": "float32"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + values.tobytes()).hexdigest()


class TelemetryAuthenticator:
    """HMAC integrity at a trusted acquisition boundary.

    Applying this to a downloaded dataset protects later pipeline transfers; it cannot
    prove that the original plant telemetry was authentic before ingestion.
    """

    def __init__(self, secret_key: bytes):
        if len(secret_key) < 16:
            raise ValueError("Use at least 16 random bytes for the acquisition key")
        self._key = bytes(secret_key)

    def sign(self, window: np.ndarray, event_id: str) -> str:
        digest = window_digest(window, event_id).encode("ascii")
        return hmac.new(self._key, digest, hashlib.sha256).hexdigest()

    def verify(self, window: np.ndarray, event_id: str, signature: str) -> bool:
        expected = self.sign(window, event_id)
        return hmac.compare_digest(expected, str(signature))
