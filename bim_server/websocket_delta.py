"""Differential IFC encoder — only transmits changed props.
Paper: Section V-A, mean payload 340 bytes vs 4.2 MB full IFC.
"""
import json, hashlib, time

class DifferentialEncoder:
    def __init__(self):
        self._hashes = {}

    def encode(self, guid, pset, props):
        key = f"{guid}::{pset}"
        h   = hashlib.md5(json.dumps(props, sort_keys=True).encode()).hexdigest()
        if self._hashes.get(key) == h:
            return None
        self._hashes[key] = h
        return json.dumps({"guid":guid,"pset":pset,"props":props,"ts":time.time()}).encode()
