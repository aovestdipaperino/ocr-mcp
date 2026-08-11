// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Building and parsing the OpenAI-compatible `/v1/chat/completions` exchange.
//!
//! Deliberately free of process management and of the HTTP call itself, so
//! every wire-format detail is testable against recorded bodies.

use crate::image_input::{DecodedImage, to_data_uri};
use crate::ocr::OcrError;
use serde_json::{Value, json};

/// Builds the request body: one text part, one image part.
#[must_use]
pub fn build_request(image: &DecodedImage, prompt: &str, max_tokens: u32) -> Value {
    json!({
        "model": "ocr",
        "stream": false,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": prompt },
                { "type": "image_url", "image_url": { "url": to_data_uri(image) } }
            ]
        }]
    })
}

/// Extracts the assistant text from a response body.
///
/// # Errors
/// Returns [`OcrError::BadResponse`] when the body is not JSON, carries a
/// server-side error object, or yields no non-blank text.
pub fn parse_response(body: &str) -> Result<String, OcrError> {
    let v: Value = serde_json::from_str(body)
        .map_err(|e| OcrError::BadResponse(format!("response was not JSON: {e}")))?;

    if let Some(msg) = v["error"]["message"].as_str() {
        return Err(OcrError::BadResponse(format!("llama-server said: {msg}")));
    }

    let text = v["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| OcrError::BadResponse("no choices in the response".to_string()))?;

    if text.trim().is_empty() {
        return Err(OcrError::BadResponse(
            "the transcription came back empty".to_string(),
        ));
    }
    Ok(text.to_string())
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
    fn request_carries_one_text_part_and_one_image_part() {
        let req = build_request(&img(), "Transcribe.", 512);
        assert_eq!(req["max_tokens"], 512);
        assert_eq!(req["stream"], false);
        let content = req["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Transcribe.");
        assert_eq!(content[1]["type"], "image_url");
        let url = content[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"), "got: {url}");
        assert_eq!(url, to_data_uri(&img()));
    }

    #[test]
    fn response_text_is_extracted() {
        let body =
            r#"{"choices":[{"message":{"role":"assistant","content":"line one\nline two"}}]}"#;
        assert_eq!(parse_response(body).unwrap(), "line one\nline two");
    }

    #[test]
    fn empty_and_malformed_responses_are_errors() {
        assert!(parse_response("not json at all").is_err());
        assert!(parse_response(r#"{"choices":[]}"#).is_err());
        assert!(parse_response(r#"{"choices":[{"message":{"content":""}}]}"#).is_err());
        assert!(parse_response(r#"{"choices":[{"message":{"content":"   "}}]}"#).is_err());
    }

    /// llama-server reports its own failures in an `error` object; that text
    /// is far more useful than "no choices".
    #[test]
    fn server_side_error_object_is_surfaced() {
        let body = r#"{"error":{"message":"failed to load image","code":500}}"#;
        let err = parse_response(body).unwrap_err();
        assert!(
            err.to_string().contains("failed to load image"),
            "got: {err}"
        );
    }

    /// A truncated generation still has usable text; it must not be discarded.
    #[test]
    fn truncated_response_still_yields_its_text() {
        let body = r#"{"choices":[{"finish_reason":"length","message":{"content":"partial tr"}}]}"#;
        assert_eq!(parse_response(body).unwrap(), "partial tr");
    }
}
