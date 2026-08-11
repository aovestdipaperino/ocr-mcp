<img src="https://raw.githubusercontent.com/aovestdipaperino/ocr-mcp/main/assets/logo.png" alt="ocr-mcp" width="128" align="right">

# ocr-mcp

[![crates.io](https://img.shields.io/crates/v/ocr-mcp.svg)](https://crates.io/crates/ocr-mcp)
[![docs.rs](https://img.shields.io/docsrs/ocr-mcp)](https://docs.rs/ocr-mcp)
[![licence](https://img.shields.io/crates/l/ocr-mcp.svg)](LICENSE)

An [MCP](https://modelcontextprotocol.io) server that transcribes screenshots
with a local [GLM-OCR](https://huggingface.co/ggml-org/GLM-OCR-GGUF) 0.9B model.
No cloud, no API key, no image ever leaving the machine.

Point any MCP host at it and the model gains one tool, `transcribe_image`. Paste
a screenshot of a stack trace into your agent and it can finally read it.

The weights are about 1.34 GiB and `llama-server` does the inference. This
server spawns that process on the first tool call, keeps it warm between calls,
and a reaper thread kills it after five idle minutes so the memory goes back to
whatever else you are running. That last part matters more than it sounds: the
whole reason this is a separate process rather than a linked library is that its
memory can be handed back.

## Install

```sh
brew install llama.cpp     # provides llama-server; any recent build works
cargo install ocr-mcp
```

`llama-server` is not vendored. It is looked up on `PATH`, then at
`OCR_MCP_BINARY`. If it is missing, the tool says so and names the fix rather
than failing obscurely.

## Register it with a host

Add this to your host's MCP config: `~/.plank/.mcp.json` or `./.mcp.json` for
plank, `~/.claude.json` for Claude Code, `mcp.json` for other hosts.

```json
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
```

## Getting the weights

Two files are needed, the model and its multimodal projector. They must be a
matched pair: a mismatched projector produces a model that loads happily and
then transcribes nonsense.

With `OCR_MCP_AUTO_DOWNLOAD=1`, the server fetches both the first time you ask
it to transcribe something. It does this lazily, immediately before spawning
`llama-server`, so startup itself is never delayed and the host's handshake is
always answered promptly. The first transcription takes as long as the download
does; every later one does not.

Without that variable the server never downloads anything, and the first
transcription returns an error naming the two files, their size, and where to
put them. An MCP server has no way to ask for consent, so it does not spend a
gigabyte of someone's bandwidth on the strength of a tool call.

To fetch them by hand instead:

```sh
mkdir -p ~/.cache/ocr-mcp && cd ~/.cache/ocr-mcp
BASE=https://huggingface.co/ggml-org/GLM-OCR-GGUF/resolve/main
curl -L "$BASE/GLM-OCR-Q8_0.gguf"        -o glm-ocr.gguf          # 906 MiB
curl -L "$BASE/mmproj-GLM-OCR-Q8_0.gguf" -o glm-ocr-mmproj.gguf   # 462 MiB
```

## The tool

`transcribe_image` takes `image`, either a local filesystem path or a
`data:image/png;base64,…` URI, and an optional `question`. With no question it
transcribes everything visible and describes the layout in a sentence.

The path is the fast route when the host shares a filesystem with the server,
which is the normal case for a stdio server. The data URI keeps the tool
correct for hosts that do not.

This is an OCR model, not a general vision model. It reads text well. Ask it
whether a user interface looks right and it will answer confidently and wrongly,
which is why the tool description tells the calling model not to.

## Configuration

Everything is environment variables, because that is the only configuration
channel every MCP host offers.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OCR_MCP_MODEL_DIR` | `~/.cache/ocr-mcp` | Where weights live |
| `OCR_MCP_MODEL` | `<dir>/glm-ocr.gguf` | Model weights |
| `OCR_MCP_MMPROJ` | `<dir>/glm-ocr-mmproj.gguf` | Projector |
| `OCR_MCP_BINARY` | *(PATH lookup)* | Explicit `llama-server` path |
| `OCR_MCP_IDLE_SECS` | `300` | Kill the child after this long unused |
| `OCR_MCP_MAX_TOKENS` | `1024` | Cap on transcription length |
| `OCR_MCP_AUTO_DOWNLOAD` | unset | Fetch missing weights on the first transcription |

Pointing `OCR_MCP_MODEL` and `OCR_MCP_MMPROJ` at a different
llama.cpp-supported vision model works, and is the supported way to trade size
for capability. Nothing in the code is specific to GLM-OCR beyond the defaults
and one mandatory flag: this model does not currently work with flash attention
enabled, so the server always passes `--flash-attn off`.

## Accuracy notes

**Images are capped at 1.75 megapixels**, and that ceiling matters more than
resolution does. Image tokens scale with pixel count, and past roughly 2700 of
them the model stops transcribing and starts confabulating: fluent text with no
relation to the image, and nothing to signal that it happened. An oversized
image also poisons the `llama-server` child, so every later request in the
session returns garbage too. The cap makes both impossible, and you should not
need to think about it.

Resolution below the cap has proved far less fragile than expected. A 2322x1268
screenshot transcribed correctly all the way down to 0.55 MP. Sharpness helps
on dense small type, but a blurry capture is not the first thing to suspect.

Images carrying an alpha channel are flattened onto white before they are sent.
macOS window captures routinely carry transparency in their rounded corners and
drop shadows, and a print-to-PDF-to-PNG pipeline can leave the real content in
the alpha channel with RGB pinned near black. Sent as-is that reads to the model
as a page of near-black pixels, and the result is a fluent, entirely wrong
transcription rather than a visible failure.

Light-on-dark terminal captures are widely reported to degrade OCR at this model
size, since the training data is overwhelmingly dark-on-light documents. Dark
screenshots transcribe correctly here once they are within the cap, so treat
this as a last resort rather than a first explanation.

## Development

```sh
cargo test --lib                              # 46 tests, no model or network needed
cargo clippy --all-targets -- -D warnings
```

The whole test suite runs against `ocr::StubOcr`, so it needs no GGUF, no
`llama-server`, and no network. That is the reason the `OcrEngine` trait exists.

To exercise the real path by hand:

```sh
printf '%s\n%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"transcribe_image","arguments":{"image":"/tmp/shot.png"}}}' \
  | ocr-mcp
```

Exactly two JSON-RPC frames should come back on stdout, and nothing else ever
should: stdout is the transport, so every diagnostic goes to stderr.

## Licence

MIT © 2026 Enzo Lombardi
