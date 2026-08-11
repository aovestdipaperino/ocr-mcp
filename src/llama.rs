// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The real backend: a lazily spawned `llama-server` child plus the HTTP call.
//!
//! Everything fiddly lives in `client.rs` and `server.rs`; this file is only
//! the lifecycle glue between them.

use crate::client;
use crate::config::Config;
use crate::image_input::DecodedImage;
use crate::ocr::{OcrEngine, OcrError};
use crate::server::Child;
use std::time::{Duration, Instant};

/// How long to wait for one transcription before giving up on the child.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(180);

/// An `OcrEngine` backed by a `llama-server` child this struct owns.
pub struct LlamaOcr {
    cfg: Config,
    child: Option<Child>,
}

impl LlamaOcr {
    /// Builds the engine. Nothing is spawned until the first `describe`.
    #[must_use]
    pub fn new(cfg: Config) -> Self {
        Self { cfg, child: None }
    }

    /// Drops the child if it has gone unused past the idle timeout, so the
    /// weights stop occupying RAM in an otherwise idle session.
    pub fn reap_if_idle(&mut self) {
        if self
            .child
            .as_ref()
            .is_some_and(|c| c.idle_expired(Instant::now()))
        {
            self.child = None;
            eprintln!("ocr-mcp: llama-server idle, shut down");
        }
    }
}

impl OcrEngine for LlamaOcr {
    fn describe(&mut self, image: &DecodedImage, prompt: &str) -> Result<String, OcrError> {
        self.reap_if_idle();
        if self.child.is_none() {
            self.child = Some(Child::spawn(&self.cfg)?);
        }
        // Unwrap-free: the branch above guarantees it.
        let Some(child) = self.child.as_mut() else {
            return Err(OcrError::SpawnFailed("the child vanished".to_string()));
        };
        child.touch();
        let url = format!("{}/v1/chat/completions", child.base_url());
        let body = client::build_request(image, prompt, self.cfg.max_tokens);

        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(REQUEST_TIMEOUT))
            .build()
            .into();
        let outcome = agent
            .post(&url)
            .send_json(&body)
            .and_then(|mut r| r.body_mut().read_to_string());

        match outcome {
            Ok(text) => client::parse_response(&text),
            Err(e) => {
                // A wedged child is worse than no child: drop it so the next
                // call starts from a clean server.
                self.child = None;
                Err(OcrError::RequestFailed(e.to_string()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config {
            model: std::path::PathBuf::from("/definitely/missing/glm-ocr.gguf"),
            mmproj: std::path::PathBuf::from("/definitely/missing/mmproj.gguf"),
            binary: None,
            idle_secs: 300,
            max_tokens: 1024,
            auto_download: false,
        }
    }

    /// With no weights on disk, the very first call must fail cleanly with the
    /// actionable error rather than trying to spawn anything.
    #[test]
    fn missing_weights_fail_before_any_process_work() {
        let mut e = LlamaOcr::new(cfg());
        let img = DecodedImage {
            media_type: "image/png",
            bytes: vec![0x89, b'P', b'N', b'G'],
        };
        let err = e.describe(&img, "Transcribe.").unwrap_err();
        assert!(matches!(err, OcrError::WeightsMissing { .. }));
        assert!(err.to_string().contains("OCR_MCP_AUTO_DOWNLOAD"));
    }
}
