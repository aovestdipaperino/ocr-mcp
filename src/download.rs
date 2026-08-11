// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Fetching the GLM-OCR weights, strictly opt-in.
//!
//! An MCP server has no UI and no way to ask for consent, so it never
//! downloads a gigabyte just because a model called a tool. Nothing here runs
//! unless `OCR_MCP_AUTO_DOWNLOAD` is set; without it, the missing-weights tool
//! error explains what to fetch.

use crate::config::Config;
use std::io::Write as _;
use std::path::Path;

/// The quantized GLM-OCR weights, `Q8_0` (the plan's original `Q4_K_M` pair
/// does not exist in this repo; the brief's URLs 404'd). Confirmed combined
/// download size for this pair: ~906 MiB + ~462 MiB = ~1.34 GiB (1,434,837,056
/// bytes), via `curl -sIL <url> | grep -i content-length`.
pub const MODEL_URL: &str =
    "https://huggingface.co/ggml-org/GLM-OCR-GGUF/resolve/main/GLM-OCR-Q8_0.gguf";
/// The companion multimodal projector. A mismatched projector produces a model
/// that loads and then transcribes nonsense, so this pairs with `MODEL_URL`.
pub const MMPROJ_URL: &str =
    "https://huggingface.co/ggml-org/GLM-OCR-GGUF/resolve/main/mmproj-GLM-OCR-Q8_0.gguf";

/// Approximate download sizes for the two files above, for the pre-fetch
/// warning in `fetch`. Not enforced or verified at runtime — just a heads-up.
const MODEL_SIZE_MIB: u64 = 907;
const MMPROJ_SIZE_MIB: u64 = 462;

/// Fetches one file to `dest`, via a `.part` file so an interrupted download
/// never leaves a truncated GGUF that looks complete.
fn fetch(url: &str, dest: &Path, approx_size_mib: u64) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let part = dest.with_extension("part");
    eprintln!("ocr-mcp: downloading {url} (~{approx_size_mib} MiB) -- ctrl-c now to abort");

    let mut resp = ureq::get(url).call().map_err(|e| format!("{url}: {e}"))?;
    let mut reader = resp.body_mut().as_reader();
    let mut file = std::fs::File::create(&part).map_err(|e| format!("{}: {e}", part.display()))?;
    let copied = std::io::copy(&mut reader, &mut file).map_err(|e| format!("{url}: {e}"))?;
    file.flush()
        .map_err(|e| format!("{}: {e}", part.display()))?;
    drop(file);

    std::fs::rename(&part, dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    eprintln!("ocr-mcp: wrote {} ({copied} bytes)", dest.display());
    Ok(())
}

/// Ensures both weight files are present, fetching what is missing.
///
/// Failures are reported on stderr and otherwise ignored: the file stays
/// absent, and the next tool call returns the missing-weights error, which is
/// a far better place to explain the problem than a startup crash.
pub fn ensure_weights(cfg: &Config) {
    for (url, dest, approx_size_mib) in [
        (MODEL_URL, &cfg.model, MODEL_SIZE_MIB),
        (MMPROJ_URL, &cfg.mmproj, MMPROJ_SIZE_MIB),
    ] {
        if dest.exists() {
            continue;
        }
        if let Err(e) = fetch(url, dest, approx_size_mib) {
            eprintln!("ocr-mcp: download failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The no-op path must be exercisable with no network: two files that
    /// already exist mean nothing is fetched.
    #[test]
    fn existing_files_are_left_alone() {
        let dir = std::env::temp_dir().join(format!("ocr-mcp-dl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("m.gguf");
        let mmproj = dir.join("p.gguf");
        std::fs::write(&model, b"weights").unwrap();
        std::fs::write(&mmproj, b"projector").unwrap();

        let cfg = Config {
            model: model.clone(),
            mmproj: mmproj.clone(),
            binary: None,
            idle_secs: 300,
            max_tokens: 1024,
            auto_download: true,
        };
        ensure_weights(&cfg);
        assert_eq!(std::fs::read(&model).unwrap(), b"weights");
        assert_eq!(std::fs::read(&mmproj).unwrap(), b"projector");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn urls_are_absolute_https() {
        assert!(MODEL_URL.starts_with("https://"));
        assert!(MMPROJ_URL.starts_with("https://"));
    }
}
