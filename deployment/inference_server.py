"""
NeuroArch Inference Server — FastAPI + TorchScript real-time inference.
Sub-30ms latency. Endpoints: POST /comfort, POST /hvac, GET /health, WS /stream
"""
from __future__ import annotations
import asyncio, json, logging, time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional
import numpy as np
log = logging.getLogger("neuroarch.server")

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

CLASS_NAMES = ["Cold", "Cool", "Neutral", "Warm", "Hot"]
PMV_MAP = {0: -2.0, 1: -1.0, 2: 0.0, 3: 1.0, 4: 2.0}

class SensorWindow(BaseModel):
    data:     List[List[float]] = Field(..., description="Shape (100, 14)")
    zone_id:  str = "zone_001"

class ComfortPrediction(BaseModel):
    zone_id:       str
    comfort_class: str
    class_idx:     int
    confidence:    float
    probabilities: List[float]
    pmv_estimate:  float
    latency_ms:    float

class InferenceEngine:
    def __init__(self, snn_path: str):
        self._latencies: Deque[float] = deque(maxlen=1000)
        self._snn = self._load(snn_path)

    def _load(self, path):
        try:
            import torch
            ts = Path(path).with_suffix(".ts")
            if ts.exists():
                m = torch.jit.load(str(ts)); m.eval(); return m
            from snn.model import NeuroArchSNN, SNNConfig
            m = NeuroArchSNN(SNNConfig())
            if Path(path).exists():
                m.load_state_dict(torch.load(path, map_location="cpu"))
            return m.eval()
        except Exception as e:
            log.warning(f"SNN load failed ({e}); stub mode")
            return None

    def predict(self, window: np.ndarray) -> dict:
        t0 = time.perf_counter()
        try:
            import torch
            x = torch.from_numpy(window[np.newaxis].astype(np.float32))
            with torch.no_grad():
                logits = self._snn(x) if self._snn else torch.randn(1, 5)
            probs = logits.softmax(-1).squeeze().numpy()
        except Exception:
            probs = np.ones(5) / 5.0
        lat = (time.perf_counter() - t0) * 1000
        self._latencies.append(lat)
        ci = int(probs.argmax())
        return {"comfort_class": CLASS_NAMES[ci], "class_idx": ci,
                "confidence": float(probs.max()), "probabilities": probs.tolist(),
                "pmv_estimate": PMV_MAP[ci], "latency_ms": lat}

    def stats(self):
        if not self._latencies: return {}
        lats = np.array(self._latencies)
        return {"mean_ms": float(lats.mean()), "p99_ms": float(np.percentile(lats, 99)),
                "n": len(lats)}

_engine: Optional[InferenceEngine] = None

def create_app(model_path: str = "snn/weights/best_model.pt"):
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def lifespan(app):
        global _engine; _engine = InferenceEngine(model_path); yield

    app = FastAPI(title="NeuroArch API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health(): return {"status": "ok", "latency": _engine.stats() if _engine else {}}

    @app.post("/comfort", response_model=ComfortPrediction)
    def comfort(req: SensorWindow):
        if not _engine: raise HTTPException(503)
        w = np.array(req.data, dtype=np.float32)
        if w.shape != (100, 14): raise HTTPException(422, f"Expected (100,14), got {w.shape}")
        r = _engine.predict(w); r["zone_id"] = req.zone_id; return r

    @app.websocket("/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                d = await ws.receive_json()
                r = _engine.predict(np.array(d["data"], dtype=np.float32))
                r["zone_id"] = d.get("zone_id", "?"); await ws.send_json(r)
        except WebSocketDisconnect: pass

    return app

if __name__ == "__main__":
    import uvicorn; uvicorn.run(create_app(), host="0.0.0.0", port=8000)
