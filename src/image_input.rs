// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Decoding and normalizing the `image` tool argument.
//!
//! The argument may be a local filesystem path (the fast path, when the host
//! shares a filesystem with this process) or a `data:` URI (correct for any
//! host that does not). Both normalize through one path so there is a single
//! set of size and format rules.

use base64::Engine as _;

/// The long edge is capped here before base64 encoding, so a 5K screenshot
/// does not become a multi-megabyte JSON body.
const MAX_LONG_EDGE: u32 = 2600;
/// Monospace text below roughly this width is where 1B-class OCR falls apart,
/// so the short edge is never reduced past it.
const MIN_SHORT_EDGE: u32 = 1600;

/// An image ready to encode into a request body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    /// MIME type detected from magic bytes.
    pub media_type: &'static str,
    /// Image bytes, possibly re-encoded after downscaling.
    pub bytes: Vec<u8>,
}

/// Detects the media type from magic bytes; `None` when not supported.
#[must_use]
pub fn detect_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG") {
        Some("image/png")
    } else if bytes.starts_with(b"\xFF\xD8\xFF") {
        Some("image/jpeg")
    } else {
        None
    }
}

/// Chooses output dimensions for `(w, h)`.
///
/// The long edge is capped at `MAX_LONG_EDGE`, but never at the cost of
/// pushing the short edge below `MIN_SHORT_EDGE`: a tall terminal capture
/// scaled to fit its height would end up unreadably narrow, which is exactly
/// the failure this function exists to prevent. Images already small enough
/// are returned untouched — nothing is ever upscaled.
#[must_use]
fn target_size(w: u32, h: u32) -> (u32, u32) {
    let long = w.max(h);
    if long <= MAX_LONG_EDGE {
        return (w, h);
    }
    let by_long = f64::from(MAX_LONG_EDGE) / f64::from(long);
    let short = w.min(h);
    let scale = if f64::from(short) * by_long < f64::from(MIN_SHORT_EDGE) {
        // Refuse to shrink past the floor; if the short edge is already below
        // it, `min` with 1.0 leaves the image alone.
        (f64::from(MIN_SHORT_EDGE) / f64::from(short)).min(1.0)
    } else {
        by_long
    };
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let scaled = |v: u32| ((f64::from(v) * scale).round() as u32).max(1);
    (scaled(w), scaled(h))
}

/// Downscales when needed, re-encoding as PNG; returns the input unchanged if
/// anything about the decode or re-encode fails, since an oversized image is
/// better than no image.
fn normalize(media_type: &'static str, bytes: Vec<u8>) -> DecodedImage {
    let Ok(img) = image::load_from_memory(&bytes) else {
        return DecodedImage { media_type, bytes };
    };
    let (tw, th) = target_size(img.width(), img.height());
    if (tw, th) == (img.width(), img.height()) {
        return DecodedImage { media_type, bytes };
    }
    let small = img.thumbnail(tw, th);
    let mut out = std::io::Cursor::new(Vec::new());
    if small.write_to(&mut out, image::ImageFormat::Png).is_ok() {
        return DecodedImage {
            media_type: "image/png",
            bytes: out.into_inner(),
        };
    }
    DecodedImage { media_type, bytes }
}

/// Decodes the `image` tool argument: a filesystem path or a `data:` URI.
///
/// # Errors
/// Returns user-facing tool error text when the argument names no readable
/// file, is a malformed data URI, or does not carry supported image bytes.
pub fn decode(arg: &str) -> Result<DecodedImage, String> {
    let raw = if let Some(rest) = arg.strip_prefix("data:") {
        let (meta, payload) = rest.split_once(',').ok_or_else(|| {
            "malformed data URI: no comma separating metadata from payload".to_string()
        })?;
        if !meta.contains("base64") {
            return Err("malformed data URI: only ;base64 payloads are supported".to_string());
        }
        base64::engine::general_purpose::STANDARD
            .decode(payload.trim())
            .map_err(|e| format!("malformed data URI: base64 payload did not decode ({e})"))?
    } else {
        let path = std::path::Path::new(arg);
        std::fs::read(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                format!("no such file: {arg}")
            } else {
                format!("could not read {arg}: {e}")
            }
        })?
    };

    let media_type = detect_media_type(&raw)
        .ok_or_else(|| format!("not a supported image (expected PNG or JPEG): {arg}"))?;
    Ok(normalize(media_type, raw))
}

/// Encodes an image as a `data:` URI for the request body.
#[must_use]
pub fn to_data_uri(img: &DecodedImage) -> String {
    format!(
        "data:{};base64,{}",
        img.media_type,
        base64::engine::general_purpose::STANDARD.encode(&img.bytes)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1x1 PNG, the smallest real image to exercise the decode paths.
    fn tiny_png() -> Vec<u8> {
        base64::engine::general_purpose::STANDARD
            .decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
            )
            .unwrap()
    }

    #[test]
    fn magic_bytes_identify_supported_formats() {
        assert_eq!(
            detect_media_type(b"\x89PNG\r\n\x1a\n rest"),
            Some("image/png")
        );
        assert_eq!(
            detect_media_type(b"\xFF\xD8\xFF\xE0 rest"),
            Some("image/jpeg")
        );
        assert_eq!(detect_media_type(b"plain text"), None);
        assert_eq!(detect_media_type(b""), None);
    }

    #[test]
    fn path_and_data_uri_decode_to_the_same_bytes() {
        let png = tiny_png();
        let dir = std::env::temp_dir().join(format!("ocr-mcp-dec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("tiny.png");
        std::fs::write(&file, &png).unwrap();

        let from_path = decode(file.to_str().unwrap()).unwrap();
        let uri = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&png)
        );
        let from_uri = decode(&uri).unwrap();

        assert_eq!(from_path, from_uri);
        assert_eq!(from_path.media_type, "image/png");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_path_and_non_image_are_rejected_with_actionable_text() {
        let err = decode("/tmp/definitely-missing-91827.png").unwrap_err();
        assert!(err.contains("no such file"), "got: {err}");

        let dir = std::env::temp_dir().join(format!("ocr-mcp-rej-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("notes.txt");
        std::fs::write(&file, b"hello").unwrap();
        let err = decode(file.to_str().unwrap()).unwrap_err();
        assert!(err.contains("not a supported image"), "got: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_data_uri_is_rejected() {
        assert!(decode("data:image/png;base64,!!!not base64!!!").is_err());
        assert!(decode("data:text/plain,hello").is_err());
    }

    /// Downscaling happens only here, at the last moment before base64
    /// encoding. A tall, narrow capture must not have its short edge pushed
    /// below the legibility floor to satisfy the long-edge cap.
    #[test]
    fn tall_narrow_capture_keeps_its_short_edge_above_the_floor() {
        let (w, h) = target_size(1700, 9000);
        assert!(w >= MIN_SHORT_EDGE, "width {w} fell below the floor");
        assert!(h <= 9000);

        // A genuinely huge image in both axes still gets capped.
        let (w, h) = target_size(6000, 4000);
        assert!(w <= MAX_LONG_EDGE && h <= MAX_LONG_EDGE);

        // A small image is never upscaled.
        assert_eq!(target_size(800, 600), (800, 600));
    }

    #[test]
    fn data_uri_round_trips() {
        let img = DecodedImage {
            media_type: "image/png",
            bytes: tiny_png(),
        };
        let uri = to_data_uri(&img);
        assert!(uri.starts_with("data:image/png;base64,"));
        assert_eq!(decode(&uri).unwrap().bytes, img.bytes);
    }
}
