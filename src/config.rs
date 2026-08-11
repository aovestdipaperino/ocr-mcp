// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Configuration, read from the environment the MCP host provides.
//!
//! Environment variables are the only configuration channel every MCP host
//! offers, so there is deliberately no config file.

use std::path::PathBuf;

#[cfg(test)]
use std::collections::HashMap;

/// Resolved runtime configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Model weights (GGUF).
    pub model: PathBuf,
    /// Companion multimodal projector (GGUF).
    pub mmproj: PathBuf,
    /// Explicit `llama-server` path; `None` means look on `PATH`.
    pub binary: Option<PathBuf>,
    /// Kill the child after this many seconds unused.
    pub idle_secs: u64,
    /// Cap on transcription length.
    pub max_tokens: u32,
    /// Opt-in to fetching missing weights on first use.
    pub auto_download: bool,
}

/// Expands a leading `~/` against `$HOME`; other paths pass through.
fn expand_tilde(raw: &str, home: Option<String>) -> PathBuf {
    match (raw.strip_prefix("~/"), home) {
        (Some(rest), Some(h)) => PathBuf::from(h).join(rest),
        _ => PathBuf::from(raw),
    }
}

impl Config {
    /// Builds the configuration from a variable lookup.
    ///
    /// Taking the lookup as a parameter rather than calling `std::env` keeps
    /// this pure and testable without mutating process-global state.
    pub fn from_env(get: &dyn Fn(&str) -> Option<String>) -> Self {
        let home = get("HOME");
        let dir = get("OCR_MCP_MODEL_DIR").map_or_else(
            || expand_tilde("~/.cache/ocr-mcp", home.clone()),
            |d| expand_tilde(&d, home.clone()),
        );
        let path_or = |key: &str, default: &str| -> PathBuf {
            get(key).map_or_else(|| dir.join(default), |v| expand_tilde(&v, home.clone()))
        };
        Self {
            model: path_or("OCR_MCP_MODEL", "glm-ocr.gguf"),
            mmproj: path_or("OCR_MCP_MMPROJ", "glm-ocr-mmproj.gguf"),
            binary: get("OCR_MCP_BINARY").map(|b| expand_tilde(&b, home.clone())),
            idle_secs: get("OCR_MCP_IDLE_SECS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(300),
            max_tokens: get("OCR_MCP_MAX_TOKENS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1024),
            auto_download: get("OCR_MCP_AUTO_DOWNLOAD").is_some_and(|v| !v.is_empty()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    #[test]
    fn defaults_derive_from_model_dir() {
        let env = env_of(&[("OCR_MCP_MODEL_DIR", "/tmp/w")]);
        let cfg = Config::from_env(&|k| env.get(k).cloned());
        assert_eq!(cfg.model, PathBuf::from("/tmp/w/glm-ocr.gguf"));
        assert_eq!(cfg.mmproj, PathBuf::from("/tmp/w/glm-ocr-mmproj.gguf"));
        assert_eq!(cfg.binary, None);
        assert_eq!(cfg.idle_secs, 300);
        assert_eq!(cfg.max_tokens, 1024);
        assert!(!cfg.auto_download);
    }

    #[test]
    fn explicit_paths_win_over_model_dir() {
        let env = env_of(&[
            ("OCR_MCP_MODEL_DIR", "/tmp/w"),
            ("OCR_MCP_MODEL", "/other/m.gguf"),
            ("OCR_MCP_MMPROJ", "/other/p.gguf"),
            ("OCR_MCP_BINARY", "/usr/local/bin/llama-server"),
            ("OCR_MCP_IDLE_SECS", "60"),
            ("OCR_MCP_MAX_TOKENS", "256"),
            ("OCR_MCP_AUTO_DOWNLOAD", "1"),
        ]);
        let cfg = Config::from_env(&|k| env.get(k).cloned());
        assert_eq!(cfg.model, PathBuf::from("/other/m.gguf"));
        assert_eq!(cfg.mmproj, PathBuf::from("/other/p.gguf"));
        assert_eq!(
            cfg.binary,
            Some(PathBuf::from("/usr/local/bin/llama-server"))
        );
        assert_eq!(cfg.idle_secs, 60);
        assert_eq!(cfg.max_tokens, 256);
        assert!(cfg.auto_download);
    }

    /// A garbage numeric value must fall back to the default rather than
    /// killing the server at startup, when nothing can report the error yet.
    #[test]
    fn unparsable_numbers_fall_back_to_defaults() {
        let env = env_of(&[("OCR_MCP_IDLE_SECS", "soon"), ("OCR_MCP_MAX_TOKENS", "")]);
        let cfg = Config::from_env(&|k| env.get(k).cloned());
        assert_eq!(cfg.idle_secs, 300);
        assert_eq!(cfg.max_tokens, 1024);
    }
}
