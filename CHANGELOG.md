# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-11

First release.

### Added

- `transcribe_image` MCP tool, taking a filesystem path or a `data:` URI and an
  optional question, served over newline-delimited JSON-RPC on stdio.
- Lazily spawned `llama-server` child, reused between calls and killed by a
  reaper thread after `OCR_MCP_IDLE_SECS` (default 300) of inactivity, plus on
  `Drop` so no orphan survives a crash.
- Opt-in weight download under `OCR_MCP_AUTO_DOWNLOAD`, fetching the GLM-OCR
  model and its matching projector to `OCR_MCP_MODEL_DIR` via a `.part` file
  that is renamed only once complete.
- Automatic alpha flattening onto white, and an area cap on the image sent to
  the model.

### Notes on two decisions that look arbitrary

`--flash-attn off` is passed unconditionally because GLM-OCR does not currently
work under `llama-server` with flash attention enabled.

`--ctx-size 12000 --parallel 1` is passed for memory rather than speed. Left to
its defaults `llama-server` reserves KV for the model's full 131072-token
context across four slots: 9.78 GB resident for a 1.4 GB model, held for the
whole idle window. Bounding both measured 2.06 GB with identical output and
unchanged latency. On a machine also hosting a large local model, the difference
is enough to push that model into swap.
