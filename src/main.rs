// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Entry point. Every module lives in the library crate; this binary only
//! wires configuration to the stdio loop.
//!
//! Invariant: stdout carries JSON-RPC frames and nothing else.

use ocr_mcp::{config::Config, llama, mcp};

fn main() {
    let cfg = Config::from_env(&|k| std::env::var(k).ok());
    eprintln!(
        "ocr-mcp {} — model {}, projector {}",
        env!("CARGO_PKG_VERSION"),
        cfg.model.display(),
        cfg.mmproj.display()
    );

    // Nothing that could block the JSON-RPC handshake happens here: any
    // weight fetch is lazy, gated on the first tool call in `LlamaOcr::describe`.
    let mut engine = llama::LlamaOcr::new(cfg);
    let stdin = std::io::stdin().lock();
    let stdout = std::io::stdout().lock();
    mcp::serve(std::io::BufReader::new(stdin), stdout, &mut engine);
}
