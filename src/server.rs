// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Owning the `llama-server` child process.
//!
//! Argument-vector construction, binary lookup and port selection are pure and
//! tested directly; spawning is exercised separately with a fake binary. The
//! child's stdout and stderr are always piped, never inherited: this process
//! speaks JSON-RPC on stdout, and an inherited child's startup banner would
//! corrupt the framing and wedge the host.

use crate::config::Config;
use crate::ocr::OcrError;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long to wait for `/health` before giving up on a freshly spawned child.
const READY_TIMEOUT: Duration = Duration::from_mins(2);
/// Gap between `/health` polls.
const POLL_INTERVAL: Duration = Duration::from_millis(250);
/// Timeout for a single `/health` liveness probe: it must never let one slow
/// call push a poll iteration past `READY_TIMEOUT`.
const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs(2);
/// How many of the most recent lines of child output to retain for error
/// messages.
const TAIL_CAPACITY: usize = 40;

/// A bounded, thread-safe tail of the most recent lines written by the
/// child's stdout/stderr, used only to enrich error text.
#[derive(Clone, Default)]
struct OutputTail(Arc<Mutex<VecDeque<String>>>);

impl OutputTail {
    fn push(&self, line: String) {
        let mut lines = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if lines.len() >= TAIL_CAPACITY {
            lines.pop_front();
        }
        lines.push_back(line);
    }

    /// Renders the retained lines, oldest first, joined by newlines.
    fn snapshot(&self) -> String {
        let lines = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        lines.iter().cloned().collect::<Vec<_>>().join("\n")
    }

    /// Drains `reader` line by line into this tail until EOF. Runs on its
    /// own thread so the child's pipes never fill and deadlock the process.
    fn drain(self, reader: impl Read) {
        let mut reader = BufReader::new(reader);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let trimmed = line.trim_end_matches(['\n', '\r']);
                    self.push(trimmed.to_string());
                }
            }
        }
    }
}

/// Builds the `llama-server` argument vector.
///
/// `--flash-attn off` is mandatory: GLM-OCR does not currently work with
/// flash attention enabled. `--no-warmup` keeps first-spawn latency down,
/// since the very next thing that happens is a real request.
#[must_use]
pub fn build_argv(cfg: &Config, port: u16) -> Vec<String> {
    vec![
        "--model".into(),
        cfg.model.display().to_string(),
        "--mmproj".into(),
        cfg.mmproj.display().to_string(),
        "--port".into(),
        port.to_string(),
        "--host".into(),
        "127.0.0.1".into(),
        "--no-warmup".into(),
        "--flash-attn".into(),
        "off".into(),
    ]
}

/// Resolves the `llama-server` binary: the configured path first, then `PATH`.
///
/// `exists` is injected so the search is testable without touching the
/// filesystem.
///
/// # Errors
/// Returns [`OcrError::BinaryMissing`] when neither lookup finds a file.
pub fn find_binary(
    cfg: &Config,
    path_var: Option<&str>,
    exists: &dyn Fn(&Path) -> bool,
) -> Result<PathBuf, OcrError> {
    if let Some(explicit) = &cfg.binary {
        return if exists(explicit) {
            Ok(explicit.clone())
        } else {
            Err(OcrError::BinaryMissing)
        };
    }
    path_var
        .unwrap_or_default()
        .split(':')
        .filter(|d| !d.is_empty())
        .map(|d| Path::new(d).join("llama-server"))
        .find(|c| exists(c))
        .ok_or(OcrError::BinaryMissing)
}

/// Asks the OS for a free localhost port by binding and immediately dropping.
///
/// # Errors
/// Returns [`OcrError::SpawnFailed`] when no port can be bound.
pub fn free_port() -> Result<u16, OcrError> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|e| OcrError::SpawnFailed(format!("could not reserve a port: {e}")))?;
    listener
        .local_addr()
        .map(|a| a.port())
        .map_err(|e| OcrError::SpawnFailed(format!("could not read the reserved port: {e}")))
}

/// A running `llama-server`, killed on `Drop` so no orphan outlives us.
pub struct Child {
    proc: std::process::Child,
    port: u16,
    last_used: Instant,
    idle: Duration,
    tail: OutputTail,
}

impl Child {
    /// Spawns `llama-server` and waits until `/health` answers.
    ///
    /// # Errors
    /// [`OcrError::WeightsMissing`] when the GGUFs are absent,
    /// [`OcrError::BinaryMissing`] when the binary cannot be found,
    /// [`OcrError::SpawnFailed`] when the process will not start, and
    /// [`OcrError::NotReady`] when `/health` never answers in time.
    pub fn spawn(cfg: &Config) -> Result<Self, OcrError> {
        if !cfg.model.exists() || !cfg.mmproj.exists() {
            return Err(OcrError::WeightsMissing {
                model: cfg.model.clone(),
                mmproj: cfg.mmproj.clone(),
                auto_download: cfg.auto_download,
            });
        }
        let path_var = std::env::var("PATH").ok();
        let bin = find_binary(cfg, path_var.as_deref(), &|p| p.exists())?;
        let port = free_port()?;

        let mut proc = Command::new(&bin)
            .args(build_argv(cfg, port))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| OcrError::SpawnFailed(format!("{}: {e}", bin.display())))?;

        let tail = OutputTail::default();
        if let Some(stdout) = proc.stdout.take() {
            let tail = tail.clone();
            std::thread::spawn(move || tail.drain(stdout));
        }
        if let Some(stderr) = proc.stderr.take() {
            let tail = tail.clone();
            std::thread::spawn(move || tail.drain(stderr));
        }

        let mut child = Self {
            proc,
            port,
            last_used: Instant::now(),
            idle: Duration::from_secs(cfg.idle_secs),
            tail,
        };
        child.await_ready()?;
        Ok(child)
    }

    /// Polls `/health` until it answers or the timeout expires.
    fn await_ready(&mut self) -> Result<(), OcrError> {
        let url = format!("{}/health", self.base_url());
        let deadline = Instant::now() + READY_TIMEOUT;
        while Instant::now() < deadline {
            if let Some(status) = self.proc.try_wait().ok().flatten() {
                return Err(OcrError::NotReady(format!(
                    "the process exited early with {status}\n{}",
                    self.tail.snapshot()
                )));
            }
            if ureq::get(&url)
                .config()
                .timeout_global(Some(HEALTH_PROBE_TIMEOUT))
                .build()
                .call()
                .is_ok()
            {
                return Ok(());
            }
            std::thread::sleep(POLL_INTERVAL);
        }
        Err(OcrError::NotReady(format!(
            "no answer from {url} after {}s\n{}",
            READY_TIMEOUT.as_secs(),
            self.tail.snapshot()
        )))
    }

    /// The `http://127.0.0.1:<port>` root this child serves.
    #[must_use]
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Records that the child was just used, restarting the idle countdown.
    pub fn touch(&mut self) {
        self.last_used = Instant::now();
    }

    /// True when the child has gone unused for longer than the idle timeout.
    #[must_use]
    pub fn idle_expired(&self, now: Instant) -> bool {
        expired(self.last_used, self.idle, now)
    }
}

/// The idle-expiry rule: has `idle` elapsed since `last_used`, as of `now`?
/// Pulled out of [`Child::idle_expired`] so it can be exercised without a
/// process.
fn expired(last_used: Instant, idle: Duration, now: Instant) -> bool {
    now.duration_since(last_used) >= idle
}

impl Drop for Child {
    fn drop(&mut self) {
        let _ = self.proc.kill();
        let _ = self.proc.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config {
            model: PathBuf::from("/w/glm-ocr.gguf"),
            mmproj: PathBuf::from("/w/glm-ocr-mmproj.gguf"),
            binary: None,
            idle_secs: 300,
            max_tokens: 1024,
            auto_download: false,
        }
    }

    /// GLM-OCR does not work under llama-server with flash attention on. This
    /// assertion is the whole reason the argv is a pure function.
    #[test]
    fn argv_disables_flash_attention() {
        let argv = build_argv(&cfg(), 8123);
        let joined = argv.join(" ");
        assert!(joined.contains("--flash-attn off"), "got: {joined}");
    }

    #[test]
    fn argv_names_model_projector_and_port() {
        let argv = build_argv(&cfg(), 8123);
        let pair = |flag: &str| {
            argv.iter()
                .position(|a| a == flag)
                .map(|i| argv[i + 1].clone())
        };
        assert_eq!(pair("--model").as_deref(), Some("/w/glm-ocr.gguf"));
        assert_eq!(pair("--mmproj").as_deref(), Some("/w/glm-ocr-mmproj.gguf"));
        assert_eq!(pair("--port").as_deref(), Some("8123"));
        assert!(argv.iter().any(|a| a == "--no-warmup"));
    }

    #[test]
    fn configured_binary_wins_over_path() {
        let mut c = cfg();
        c.binary = Some(PathBuf::from("/opt/llama-server"));
        let found = find_binary(&c, Some("/usr/bin:/bin"), &|p| {
            p == Path::new("/opt/llama-server")
        });
        assert_eq!(found.unwrap(), PathBuf::from("/opt/llama-server"));
    }

    #[test]
    fn path_is_searched_when_no_binary_configured() {
        let found = find_binary(&cfg(), Some("/usr/bin:/opt/bin"), &|p| {
            p == Path::new("/opt/bin/llama-server")
        });
        assert_eq!(found.unwrap(), PathBuf::from("/opt/bin/llama-server"));
    }

    #[test]
    fn missing_binary_is_an_error() {
        let e = find_binary(&cfg(), Some("/usr/bin"), &|_| false).unwrap_err();
        assert_eq!(e, OcrError::BinaryMissing);

        let mut c = cfg();
        c.binary = Some(PathBuf::from("/nope/llama-server"));
        assert_eq!(
            find_binary(&c, None, &|_| false).unwrap_err(),
            OcrError::BinaryMissing
        );
    }

    #[test]
    fn free_port_is_usable_and_nonzero() {
        let p = free_port().unwrap();
        assert!(p > 0);
        // Binding it again must succeed: the probe socket has been dropped.
        assert!(std::net::TcpListener::bind(("127.0.0.1", p)).is_ok());
    }

    /// The idle rule is arithmetic on an injected `now`, so it needs no
    /// process and no sleeping.
    #[test]
    fn idle_expiry_is_measured_from_the_last_use() {
        let idle = Duration::from_mins(5);
        let last_used = Instant::now();
        let at = |elapsed: Duration| last_used + elapsed;
        assert!(!expired(last_used, idle, at(Duration::from_secs(299))));
        assert!(expired(last_used, idle, at(Duration::from_mins(5))));
        assert!(expired(last_used, idle, at(Duration::from_hours(1))));
    }

    /// Pushing more lines than the cap must keep only the most recent ones,
    /// in order, with no process involved.
    #[test]
    fn output_tail_keeps_only_the_most_recent_lines() {
        let tail = OutputTail::default();
        for i in 0..(TAIL_CAPACITY + 10) {
            tail.push(format!("line {i}"));
        }
        let snapshot = tail.snapshot();
        let lines: Vec<&str> = snapshot.lines().collect();
        assert_eq!(lines.len(), TAIL_CAPACITY);
        assert_eq!(lines.first(), Some(&"line 10"));
        assert_eq!(
            lines.last().copied(),
            Some(format!("line {}", TAIL_CAPACITY + 9)).as_deref()
        );
    }
}
