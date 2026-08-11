// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The real backend: a lazily spawned `llama-server` child plus the HTTP call.
//!
//! Everything fiddly lives in `client.rs` and `server.rs`; this file is only
//! the lifecycle glue between them.

use crate::client;
use crate::config::Config;
use crate::download;
use crate::image_input::DecodedImage;
use crate::ocr::{OcrEngine, OcrError};
use crate::server::Child;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long to wait for one transcription before giving up on the child.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(180);

/// How often the reaper thread wakes to check for an idle child. Small
/// enough that the idle timeout is honoured promptly, large enough that the
/// thread spends essentially all of its life parked rather than spinning.
const REAP_TICK: Duration = Duration::from_secs(5);

/// A background thread that periodically drops the shared child once it has
/// gone idle. Stopping it (on `Drop`) is immediate: it parks on a channel
/// `recv_timeout` rather than sleeping, so sending the stop signal wakes it
/// right away instead of waiting out the next tick.
struct Reaper {
    stop: mpsc::Sender<()>,
    handle: Option<JoinHandle<()>>,
}

impl Reaper {
    fn spawn(child: Arc<Mutex<Option<Child>>>) -> Self {
        let (stop, rx) = mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            loop {
                match rx.recv_timeout(REAP_TICK) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        let now = Instant::now();
                        let mut guard = child.lock().unwrap_or_else(PoisonError::into_inner);
                        if guard.as_ref().is_some_and(|c| c.should_reap(now)) {
                            *guard = None;
                            eprintln!("ocr-mcp: llama-server idle, shut down");
                        }
                    }
                }
            }
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for Reaper {
    fn drop(&mut self) {
        // The send only fails if the thread already exited; either way we
        // still join so no thread outlives this struct.
        let _ = self.stop.send(());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// An `OcrEngine` backed by a `llama-server` child this struct owns.
///
/// The child lives behind `Arc<Mutex<..>>` because both `describe` and the
/// background [`Reaper`] touch it: `describe` to spawn/use/tear it down, the
/// reaper to drop it once idle. The lock is never held across the blocking
/// HTTP request — only to check/set state — so a long transcription never
/// blocks the reaper thread, and a `busy` flag (set for the request's
/// duration) stops the reaper from ever winning a race against it.
pub struct LlamaOcr {
    cfg: Config,
    child: Arc<Mutex<Option<Child>>>,
    _reaper: Reaper,
}

impl LlamaOcr {
    /// Builds the engine. Nothing is spawned until the first `describe`; the
    /// reaper thread starts immediately but does nothing while `child` is
    /// `None`.
    #[must_use]
    pub fn new(cfg: Config) -> Self {
        let child = Arc::new(Mutex::new(None));
        let reaper = Reaper::spawn(Arc::clone(&child));
        Self {
            cfg,
            child,
            _reaper: reaper,
        }
    }
}

impl OcrEngine for LlamaOcr {
    fn describe(&mut self, image: &DecodedImage, prompt: &str) -> Result<String, OcrError> {
        // Ensure a live, non-idle child, mark it busy, and grab what the
        // request needs — all under the lock, none of it blocking.
        let base_url = {
            let mut guard = self.child.lock().unwrap_or_else(PoisonError::into_inner);
            if guard
                .as_ref()
                .is_some_and(|c| c.should_reap(Instant::now()))
            {
                *guard = None;
            }
            if guard.is_none() {
                // Gated on auto_download inside ensure_weights's caller
                // contract: only fetch when the user opted in. A file that
                // already exists short-circuits, so a second call here never
                // re-downloads.
                if self.cfg.auto_download {
                    download::ensure_weights(&self.cfg);
                }
                *guard = Some(Child::spawn(&self.cfg)?);
            }
            let child = guard.as_mut().expect("just ensured Some above");
            child.touch();
            child.set_busy(true);
            child.base_url()
        };

        let url = format!("{base_url}/v1/chat/completions");
        let body = client::build_request(image, prompt, self.cfg.max_tokens);

        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(REQUEST_TIMEOUT))
            .build()
            .into();
        let outcome = agent
            .post(&url)
            .send_json(&body)
            .and_then(|mut r| r.body_mut().read_to_string());

        // Re-acquire only to record completion; the HTTP call above happened
        // with the lock released.
        let mut guard = self.child.lock().unwrap_or_else(PoisonError::into_inner);
        match outcome {
            Ok(text) => {
                if let Some(child) = guard.as_mut() {
                    child.touch();
                    child.set_busy(false);
                }
                drop(guard);
                client::parse_response(&text)
            }
            Err(e) => {
                // A wedged child is worse than no child: drop it so the next
                // call starts from a clean server.
                *guard = None;
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
