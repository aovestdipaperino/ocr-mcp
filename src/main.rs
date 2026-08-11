// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! `ocr-mcp` — an MCP stdio server that transcribes images with a local
//! GLM-OCR model, spawning `llama-server` on demand.

mod config;

fn main() {
    let cfg = config::Config::from_env(&|k| std::env::var(k).ok());
    eprintln!("ocr-mcp: model={}", cfg.model.display());
}
