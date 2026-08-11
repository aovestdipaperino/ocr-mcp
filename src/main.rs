// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

use ocr_mcp::config::Config;

fn main() {
    let cfg = Config::from_env(&|k| std::env::var(k).ok());
    eprintln!("ocr-mcp: model={}", cfg.model.display());
}
