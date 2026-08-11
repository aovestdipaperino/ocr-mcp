// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Newline-delimited JSON-RPC over stdio.
//!
//! Pure request-to-response translation over an [`OcrEngine`], so the whole
//! protocol surface is tested by feeding recorded frames in and asserting the
//! frames that come out.
//!
//! Invariant: stdout carries JSON-RPC frames and nothing else. Every
//! diagnostic goes to stderr.

use crate::image_input;
use crate::ocr::{DEFAULT_PROMPT, OcrEngine};
use serde_json::{Value, json};
use std::io::{BufRead, Write};

/// The one tool this server exposes.
pub const TOOL_NAME: &str = "transcribe_image";

/// The tool schema advertised in `tools/list`.
///
/// The description is the only place the calling model learns what this model
/// is for. It says "transcribe", names both accepted input forms, and states
/// the limitation outright, because a 0.9B OCR model asked to judge a layout
/// will answer confidently and wrongly.
#[must_use]
pub fn tool_schema() -> Value {
    json!({
        "name": TOOL_NAME,
        "description": "Transcribe the text in a screenshot or document image using a local OCR model. \
    Use this for screenshots of terminals, stack traces, logs, diffs, code, and documents: it returns the \
    visible text verbatim plus a short description of the layout. It is an OCR model, not a general vision \
    model — do not use it to judge whether a user interface looks correct, to compare designs, or to reason \
    about non-textual imagery.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "A local filesystem path to a PNG or JPEG, or a data URI of the form data:image/png;base64,<payload>."
                },
                "question": {
                    "type": "string",
                    "description": "Optional. A specific question about the image's text. When omitted, the whole image is transcribed."
                }
            },
            "required": ["image"]
        }
    })
}

/// Wraps text as a successful `tools/call` result.
fn text_result(id: &Value, text: &str) -> String {
    json!({"jsonrpc":"2.0","id":id,"result":{"content":[{"type":"text","text":text}]}}).to_string()
}

/// Wraps text as a failed `tools/call` result.
///
/// Tool failures are results with `isError`, not JSON-RPC errors: the model
/// should see the message and adapt, not have the call disappear.
fn error_result(id: &Value, text: &str) -> String {
    json!({"jsonrpc":"2.0","id":id,"result":{"isError":true,"content":[{"type":"text","text":text}]}})
        .to_string()
}

/// A protocol-level error frame.
fn rpc_error(id: &Value, code: i32, message: &str) -> String {
    json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}}).to_string()
}

/// Runs one `tools/call`.
fn tools_call(id: &Value, params: &Value, engine: &mut dyn OcrEngine) -> String {
    if params["name"] != TOOL_NAME {
        return rpc_error(id, -32602, &format!("unknown tool: {}", params["name"]));
    }
    let args = &params["arguments"];
    let Some(image_arg) = args["image"].as_str() else {
        return error_result(
            id,
            "the `image` argument is required: a file path or a data: URI",
        );
    };
    let image = match image_input::decode(image_arg) {
        Ok(i) => i,
        Err(e) => return error_result(id, &e),
    };
    let prompt = args["question"].as_str().unwrap_or(DEFAULT_PROMPT);
    match engine.describe(&image, prompt) {
        Ok(text) => text_result(id, &text),
        Err(e) => error_result(id, &e.to_string()),
    }
}

/// Handles one request line, returning the response line.
///
/// Returns `None` for notifications (requests with no `id`), which by the
/// JSON-RPC contract must not be answered.
#[must_use]
pub fn handle(line: &str, engine: &mut dyn OcrEngine) -> Option<String> {
    let Ok(req) = serde_json::from_str::<Value>(line) else {
        return Some(rpc_error(
            &Value::Null,
            -32700,
            "parse error: line was not valid JSON",
        ));
    };
    let id = req.get("id").cloned();
    let method = req["method"].as_str().unwrap_or_default();
    // A notification carries no id and gets no reply, whatever it asks for.
    let id = id?;

    Some(match method {
        "initialize" => {
            let version = req["params"]["protocolVersion"]
                .as_str()
                .unwrap_or("2024-11-05");
            json!({"jsonrpc":"2.0","id":id,"result":{
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "ocr-mcp", "version": env!("CARGO_PKG_VERSION")}
            }})
            .to_string()
        }
        "tools/list" => {
            json!({"jsonrpc":"2.0","id":id,"result":{"tools":[tool_schema()]}}).to_string()
        }
        "tools/call" => tools_call(&id, &req["params"], engine),
        other => rpc_error(&id, -32601, &format!("method not found: {other}")),
    })
}

/// Reads request lines until EOF, writing one response line per request.
pub fn serve(input: impl BufRead, mut output: impl Write, engine: &mut dyn OcrEngine) {
    for line in input.lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        if let Some(reply) = handle(&line, engine) {
            if writeln!(output, "{reply}").is_err() || output.flush().is_err() {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocr::{OcrError, StubOcr};

    fn call(line: &str, engine: &mut dyn OcrEngine) -> Value {
        serde_json::from_str(&handle(line, engine).expect("expected a reply")).unwrap()
    }

    #[test]
    fn initialize_echoes_the_clients_protocol_version() {
        let mut e = StubOcr::new("ok");
        let v = call(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"plank","version":"1.0"}}}"#,
            &mut e,
        );
        assert_eq!(v["id"], 1);
        assert_eq!(v["result"]["protocolVersion"], "2024-11-05");
        assert!(v["result"]["capabilities"]["tools"].is_object());
        assert_eq!(v["result"]["serverInfo"]["name"], "ocr-mcp");
    }

    #[test]
    fn notifications_get_no_reply() {
        let mut e = StubOcr::new("ok");
        assert!(
            handle(
                r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
                &mut e
            )
            .is_none()
        );
    }

    #[test]
    fn tools_list_advertises_one_tool_with_both_input_forms() {
        let mut e = StubOcr::new("ok");
        let v = call(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#, &mut e);
        let tools = v["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], TOOL_NAME);
        let props = &tools[0]["inputSchema"]["properties"];
        assert!(props["image"].is_object());
        assert!(props["question"].is_object());
        assert_eq!(tools[0]["inputSchema"]["required"][0], "image");
        // The description must steer the caller away from asking this model
        // to interpret a UI.
        let desc = tools[0]["description"].as_str().unwrap();
        assert!(desc.to_lowercase().contains("transcri"), "got: {desc}");
    }

    fn png_path() -> std::path::PathBuf {
        use base64::Engine as _;
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let png = base64::engine::general_purpose::STANDARD
            .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
            .unwrap();
        // Each call gets its own file: parallel tests must never share a path,
        // or one test's write/rewrite can race another test's read.
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("ocr-mcp-mcp-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("shot.png");
        std::fs::write(&p, png).unwrap();
        p
    }

    #[test]
    fn tools_call_returns_the_transcription_as_text_content() {
        let mut e = StubOcr::new("error: index out of range");
        let req = json!({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": { "name": TOOL_NAME, "arguments": { "image": png_path().to_str().unwrap() } }
        });
        let v = call(&req.to_string(), &mut e);
        assert_eq!(v["result"]["content"][0]["type"], "text");
        assert_eq!(
            v["result"]["content"][0]["text"],
            "error: index out of range"
        );
        assert_ne!(v["result"]["isError"], true);
        // With no question, the default transcription prompt is used verbatim.
        assert_eq!(e.last_prompt.as_deref(), Some(DEFAULT_PROMPT));
    }

    /// The caller's question is passed through untouched — never merged into a
    /// combined instruction, which is what makes a 0.9B model start inventing.
    #[test]
    fn a_supplied_question_is_passed_through_verbatim() {
        let mut e = StubOcr::new("42");
        let req = json!({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": { "name": TOOL_NAME, "arguments": {
                "image": png_path().to_str().unwrap(), "question": "What is the exit code?" } }
        });
        let _ = call(&req.to_string(), &mut e);
        assert_eq!(e.last_prompt.as_deref(), Some("What is the exit code?"));
    }

    #[test]
    fn engine_failure_becomes_an_error_result_not_a_protocol_error() {
        let mut e = StubOcr::failing(OcrError::BinaryMissing);
        let req = json!({
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": { "name": TOOL_NAME, "arguments": { "image": png_path().to_str().unwrap() } }
        });
        let v = call(&req.to_string(), &mut e);
        assert_eq!(v["result"]["isError"], true);
        assert!(
            v["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("brew install llama.cpp")
        );
    }

    #[test]
    fn bad_image_argument_is_an_error_result() {
        let mut e = StubOcr::new("unused");
        let req = json!({
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": { "name": TOOL_NAME, "arguments": { "image": "/tmp/missing-77123.png" } }
        });
        let v = call(&req.to_string(), &mut e);
        assert_eq!(v["result"]["isError"], true);
        assert!(
            v["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("no such file")
        );

        let req = json!({
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": { "name": TOOL_NAME, "arguments": {} }
        });
        let v = call(&req.to_string(), &mut e);
        assert_eq!(v["result"]["isError"], true);
    }

    #[test]
    fn unknown_method_and_unknown_tool_are_reported_distinctly() {
        let mut e = StubOcr::new("ok");
        let v = call(
            r#"{"jsonrpc":"2.0","id":8,"method":"resources/list"}"#,
            &mut e,
        );
        assert_eq!(v["error"]["code"], -32601);

        let req = json!({
            "jsonrpc": "2.0", "id": 9, "method": "tools/call",
            "params": { "name": "nope", "arguments": {} }
        });
        let v = call(&req.to_string(), &mut e);
        assert_eq!(v["error"]["code"], -32602);
    }

    #[test]
    fn unparsable_line_yields_a_parse_error_frame() {
        let mut e = StubOcr::new("ok");
        let v = call("{ this is not json", &mut e);
        assert_eq!(v["error"]["code"], -32700);
    }

    /// The invariant that keeps stdio usable: a full session writes nothing to
    /// the output stream but well-formed frames, one per line.
    #[test]
    fn a_full_session_writes_only_json_frames() {
        let mut e = StubOcr::new("transcribed");
        let session = format!(
            "{}\n{}\n{}\n{}\n",
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#,
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
            json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
                "name": TOOL_NAME, "arguments": {"image": png_path().to_str().unwrap()}}})
        );
        let mut out: Vec<u8> = Vec::new();
        serve(std::io::Cursor::new(session), &mut out, &mut e);

        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines.len(),
            3,
            "the notification must not be answered: {text}"
        );
        for line in lines {
            let v: Value = serde_json::from_str(line).expect("every line must be a frame");
            assert_eq!(v["jsonrpc"], "2.0");
        }
    }
}
