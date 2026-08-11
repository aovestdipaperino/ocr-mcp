# ocr-mcp

An MCP server that transcribes screenshots with a local GLM-OCR 0.9B model.

It spawns `llama-server` on the first tool call, keeps it alive between calls,
and kills it after five idle minutes so the weights stop occupying RAM.

## Install

    brew install llama.cpp
    cargo install ocr-mcp

## Register it

    {
      "mcpServers": {
        "ocr": {
          "command": "ocr-mcp",
          "env": {
            "OCR_MCP_MODEL_DIR": "~/.cache/ocr-mcp",
            "OCR_MCP_AUTO_DOWNLOAD": "1"
          }
        }
      }
    }

With `OCR_MCP_AUTO_DOWNLOAD` set, the weights (Q8_0, about 1.34 GiB combined)
are fetched on first start. Without it, the server tells you what to download
and where to put it.

## The tool

`transcribe_image` takes `image` — a local path or a `data:image/png;base64,…`
URI — and an optional `question`. With no question it transcribes everything
visible and describes the layout.

This is an OCR model, not a general vision model. It reads text well; it will
confabulate if you ask it whether a user interface looks right.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OCR_MCP_MODEL_DIR` | `~/.cache/ocr-mcp` | Where weights live |
| `OCR_MCP_MODEL` | `<dir>/glm-ocr.gguf` | Model weights |
| `OCR_MCP_MMPROJ` | `<dir>/glm-ocr-mmproj.gguf` | Projector |
| `OCR_MCP_BINARY` | *(PATH lookup)* | Explicit `llama-server` path |
| `OCR_MCP_IDLE_SECS` | `300` | Kill the child after this long unused |
| `OCR_MCP_MAX_TOKENS` | `1024` | Cap on transcription length |
| `OCR_MCP_AUTO_DOWNLOAD` | unset | Fetch missing weights on first start |

Pointing `OCR_MCP_MODEL` and `OCR_MCP_MMPROJ` at a different llama.cpp-supported
vision model works, and is the supported way to trade size for capability.

## Accuracy notes

Two things matter more than the model choice. Capture at 2x: monospace text
below roughly 1600px wide defeats every model in this weight class. And prefer a
light background — light-on-dark degrades OCR measurably, because the training
data is overwhelmingly dark-on-light documents.

Images carrying an alpha channel are flattened onto a white background
automatically, for the same reason. macOS window captures routinely have
transparency in their corners and drop shadows, and a PDF-to-PNG pipeline can
end up with real content living only in the alpha channel and RGB pinned near
black. Sent as-is, that reads to the model as a page full of near-black
pixels, and it will confabulate a fluent, entirely wrong transcription rather
than fail visibly — the worst outcome an OCR tool can produce.

## Licence

MIT © 2026 Enzo Lombardi
