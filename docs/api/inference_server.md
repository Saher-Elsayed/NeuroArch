# Inference Server API

## POST /comfort
Classify comfort from a (100, 14) sensor window.
Returns: comfort_class, confidence, probabilities, pmv_estimate, latency_ms

## GET /health
Returns latency statistics.

## WebSocket /stream
Real-time streaming inference at 30 Hz.
