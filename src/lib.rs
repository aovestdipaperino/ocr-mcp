// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `ocr-mcp` — an MCP stdio server that transcribes images with a local
//! GLM-OCR model, spawning `llama-server` on demand.

pub mod client;
pub mod config;
pub mod download;
pub mod image_input;
pub mod llama;
pub mod mcp;
pub mod ocr;
pub mod server;
