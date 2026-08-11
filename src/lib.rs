// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `ocr-mcp` — an MCP stdio server that transcribes images with a local
//! GLM-OCR model, spawning `llama-server` on demand.
//!
//! The binary is the product; this library exists so the pieces can be tested
//! without a model, a GPU, or a network, and it is documented here for anyone
//! who wants to reuse or audit them.
//!
//! # How a transcription happens
//!
//! ```text
//! host --stdio--> mcp::serve --> llama::LlamaOcr --> server::Child (llama-server)
//!                     |               |                     ^
//!                 tools/call    image_input::decode     client::build_request
//! ```
//!
//! [`mcp`] speaks newline-delimited JSON-RPC over stdin and stdout and is pure
//! request-to-response translation over the [`ocr::OcrEngine`] trait.
//! [`image_input`] turns the tool's `image` argument, either a filesystem path
//! or a `data:` URI, into normalized bytes. [`llama::LlamaOcr`] is the real
//! engine: it lazily spawns a [`server::Child`], calls it through
//! [`client`], and lets a reaper thread drop the child once it has gone idle.
//! [`ocr::StubOcr`] stands in for all of that in tests.
//!
//! # Two invariants worth knowing before you change anything
//!
//! **stdout carries JSON-RPC frames and nothing else.** It is the transport.
//! Every diagnostic goes to stderr, and the `llama-server` child's own streams
//! are piped and drained rather than inherited, because an inherited child
//! writing a startup banner would corrupt the framing and wedge the host.
//!
//! **Oversized images are worse than slow.** Image tokens scale with pixel
//! area, and past roughly 2700 of them this model stops transcribing and
//! begins confabulating: fluent text with no relation to the picture, and
//! nothing to signal it happened. An oversized image also poisons the child
//! process, so every later request returns garbage too. [`image_input`] caps
//! area to make both impossible; see `MAX_PIXELS` there for the measurements.

pub mod client;
pub mod config;
pub mod download;
pub mod image_input;
pub mod llama;
pub mod mcp;
pub mod ocr;
pub mod server;
