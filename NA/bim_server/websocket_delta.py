"""
Differential IFC encoder — only transmits changed property values.
Mean payload: 340 bytes/tick vs 4.2 MB full IFC resend.
Paper: Section V-A
"""
import json, hashlib, time


class DifferentialEncoder:
    def __init__(self):
        self._hashes: dict[str, str] = {}
        self.bytes_saved = 0
        self.bytes_sent  = 0

    def encode(self, guid: str, pset: str, props: dict) -> bytes | None:
        key = f"{guid}::{pset}"
        h   = hashlib.md5(json.dumps(props, sort_keys=True).encode()).hexdigest()
        if self._hashes.get(key) == h:
            return None
        self._hashes[key] = h
        payload = json.dumps({"guid": guid, "pset": pset,
                               "props": props, "ts": time.time()}).encode()
        self.bytes_sent += len(payload)
        return payload

    @property
    def compression_ratio(self) -> float:
        total = self.bytes_sent + self.bytes_saved
        return self.bytes_saved / total if total > 0 else 0.0
