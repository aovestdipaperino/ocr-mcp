// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! The inference boundary.
//!
//! `OcrEngine` exists for the same reason plank's `Engine` trait does: it lets
//! the JSON-RPC layer be driven end to end in CI, where no GGUF and no
//! `llama-server` exist. `StubOcr` is this crate's `EchoEngine`.

use crate::image_input::DecodedImage;
use std::path::PathBuf;

/// The default instruction when the caller supplies no question.
///
/// It asks for transcription and layout only. At 0.9B, an instruction that
/// invites reasoning yields confident fiction instead of characters, so the
/// wording deliberately never asks the model to explain, diagnose, or judge.
pub const DEFAULT_PROMPT: &str = "Transcribe every piece of text visible in this image, verbatim and in reading order, preserving line breaks. Then describe the layout in one short paragraph: what regions exist and where. Do not explain, interpret, or guess at anything that is not legible; write [illegible] instead.";

/// Everything that can go wrong between a tool call and a transcription.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OcrError {
    /// No `llama-server` on `PATH` and none configured.
    BinaryMissing,
    /// Model and/or projector are absent from disk.
    WeightsMissing {
        model: PathBuf,
        mmproj: PathBuf,
        auto_download: bool,
    },
    /// The child process could not be started.
    SpawnFailed(String),
    /// The child started but never answered `/health` in time; carries the
    /// tail of its captured stderr.
    NotReady(String),
    /// The HTTP request itself failed or timed out.
    RequestFailed(String),
    /// A response arrived but was not usable.
    BadResponse(String),
}

impl std::fmt::Display for OcrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BinaryMissing => write!(
                f,
                "llama-server was not found on PATH. Install it with `brew install llama.cpp`, \
                 or set OCR_MCP_BINARY to its full path."
            ),
            Self::WeightsMissing {
                model,
                mmproj,
                auto_download,
            } => {
                write!(
                    f,
                    "the GLM-OCR weights are not installed. Expected:\n  {}\n  {}\n",
                    model.display(),
                    mmproj.display()
                )?;
                if *auto_download {
                    write!(
                        f,
                        "The automatic download failed; fetch both files by hand."
                    )
                } else {
                    write!(
                        f,
                        "Set OCR_MCP_AUTO_DOWNLOAD=1 in this server's env to fetch them \
                         (about 1 GB, once), or download both files by hand."
                    )
                }
            }
            Self::SpawnFailed(e) => write!(f, "could not start llama-server: {e}"),
            Self::NotReady(tail) => write!(
                f,
                "llama-server did not become ready in time. Its last output was:\n{tail}"
            ),
            Self::RequestFailed(e) => write!(f, "the transcription request failed: {e}"),
            Self::BadResponse(e) => write!(
                f,
                "the model returned nothing usable ({e}). If the image is a light-on-dark \
                 terminal capture, try a light background: OCR accuracy at this model size \
                 degrades measurably on inverted contrast."
            ),
        }
    }
}

/// Turns one image into text.
pub trait OcrEngine {
    /// Transcribes or answers a question about `image`.
    ///
    /// # Errors
    /// Returns the variant of [`OcrError`] describing the failure; its
    /// `Display` text is what the caller sees.
    fn describe(&mut self, image: &DecodedImage, prompt: &str) -> Result<String, OcrError>;
}

/// A backend that needs no model, no binary, and no network.
pub struct StubOcr {
    reply: Result<String, OcrError>,
    /// The prompt from the most recent call, for assertions.
    pub last_prompt: Option<String>,
}

impl StubOcr {
    /// A stub that always succeeds with `reply`.
    #[must_use]
    pub fn new(reply: &str) -> Self {
        Self {
            reply: Ok(reply.to_string()),
            last_prompt: None,
        }
    }

    /// A stub that always fails with `err`.
    #[must_use]
    pub fn failing(err: OcrError) -> Self {
        Self {
            reply: Err(err),
            last_prompt: None,
        }
    }
}

impl OcrEngine for StubOcr {
    fn describe(&mut self, _image: &DecodedImage, prompt: &str) -> Result<String, OcrError> {
        self.last_prompt = Some(prompt.to_string());
        self.reply.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img() -> DecodedImage {
        DecodedImage {
            media_type: "image/png",
            bytes: vec![0x89, b'P', b'N', b'G'],
        }
    }

    #[test]
    fn stub_returns_its_canned_reply() {
        let mut e = StubOcr::new("error: index out of range");
        assert_eq!(
            e.describe(&img(), DEFAULT_PROMPT).unwrap(),
            "error: index out of range"
        );
    }

    #[test]
    fn stub_records_the_prompt_it_was_given() {
        let mut e = StubOcr::new("ok");
        let _ = e.describe(&img(), "What is the exit code?");
        assert_eq!(e.last_prompt.as_deref(), Some("What is the exit code?"));
    }

    #[test]
    fn stub_can_be_made_to_fail() {
        let mut e = StubOcr::failing(OcrError::BinaryMissing);
        assert!(e.describe(&img(), DEFAULT_PROMPT).is_err());
    }

    /// Error text is a product surface: each variant must name the fix.
    #[test]
    fn error_text_names_the_fix() {
        assert!(
            OcrError::BinaryMissing
                .to_string()
                .contains("brew install llama.cpp")
        );

        let missing = OcrError::WeightsMissing {
            model: PathBuf::from("/w/glm-ocr.gguf"),
            mmproj: PathBuf::from("/w/glm-ocr-mmproj.gguf"),
            auto_download: false,
        };
        let text = missing.to_string();
        assert!(text.contains("/w/glm-ocr.gguf"));
        assert!(text.contains("/w/glm-ocr-mmproj.gguf"));
        assert!(text.contains("OCR_MCP_AUTO_DOWNLOAD"));

        assert!(
            OcrError::NotReady("boom".into())
                .to_string()
                .contains("boom")
        );
    }

    /// Light-on-dark degrades OCR at this size; the hint belongs in the text
    /// the user actually reads, not in a comment.
    #[test]
    fn request_failure_mentions_the_contrast_caveat() {
        let text = OcrError::BadResponse("empty".into()).to_string();
        assert!(text.contains("light-on-dark"), "got: {text}");
    }
}
