"""Prometheus metrics for NeuroArch inference server."""
from __future__ import annotations
import time
from collections import deque
from typing import Deque

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    HAS_PROM = True
except ImportError:
    HAS_PROM = False


class NeuroArchMetrics:
    """Unified metrics: Prometheus + in-process deque for latency."""

    def __init__(self, port: int = 9090):
        self._latencies: Deque[float] = deque(maxlen=10000)
        self._n_inferences = 0
        self._n_errors = 0
        self._comfort_counts = {i: 0 for i in range(5)}
        if HAS_PROM:
            self._hist = Histogram("neuroarch_inference_latency_ms",
                                   "SNN inference latency", buckets=[1,2,4,6,8,12,20,30])
            self._counter = Counter("neuroarch_inferences_total", "Total inferences")
            self._class_gauge = Gauge("neuroarch_comfort_class", "Latest comfort class")

    def record_inference(self, latency_ms: float, comfort_class: int):
        self._latencies.append(latency_ms)
        self._n_inferences += 1
        self._comfort_counts[comfort_class] = self._comfort_counts.get(comfort_class, 0) + 1
        if HAS_PROM:
            self._hist.observe(latency_ms)
            self._counter.inc()
            self._class_gauge.set(comfort_class)

    def record_error(self):
        self._n_errors += 1

    def summary(self) -> dict:
        import numpy as np
        if not self._latencies:
            return {}
        lats = np.array(self._latencies)
        return {
            "n_inferences":   self._n_inferences,
            "n_errors":       self._n_errors,
            "mean_ms":        float(lats.mean()),
            "p50_ms":         float(np.percentile(lats, 50)),
            "p95_ms":         float(np.percentile(lats, 95)),
            "p99_ms":         float(np.percentile(lats, 99)),
            "comfort_dist":   self._comfort_counts,
        }
