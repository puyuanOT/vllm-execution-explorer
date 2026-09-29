# Changelog

## 0.1.0 — 2026-09-29

- Added the progressive Configure workflow for timed request batches, execution-host setup, model and LoRA selection, quantization, and vLLM settings.
- Added the Explore workflow with the 3D execution overview, request lanes, component drill-down, inspector, and deterministic playback controls.
- Added loopback HTTP capture for instrumented vLLM 0.30.x execution, SSH GPU-host setup, operator-local artifact transfer, cancellation, and finalized recording import.
- Added host-specific capability reporting, structural export sanitization, deterministic replay checks, responsive and accessible UI behavior, and non-GPU CI.
- Completed NVIDIA A10 GPU acceptance for the supported v0.1 paths. The sanitized evidence and known limits are recorded in [`docs/acceptance`](docs/acceptance/README.md).
