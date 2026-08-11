// Copyright (c) 2026 Enzo Lombardi
// SPDX-License-Identifier: MIT

//! Decoding and normalizing the `image` tool argument.
//!
//! The argument may be a local filesystem path (the fast path, when the host
//! shares a filesystem with this process) or a `data:` URI (correct for any
//! host that does not). Both normalize through one path so there is a single
//! set of size and format rules.

use base64::Engine as _;

/// Maximum pixel area sent to the model.
///
/// The binding limit is **area**, not either edge: the vision encoder emits
/// image tokens in proportion to pixel count, and past roughly 2700 tokens
/// GLM-OCR stops transcribing and starts confabulating — fluent text with no
/// relation to the image. Worse, an oversized image poisons the `llama-server`
/// process: every subsequent request on that child returns garbage too, so one
/// large screenshot ruins a whole session.
///
/// Measured on this model: 2318x902 (2.09 MP, 2678 tokens) transcribes
/// correctly; 2322x1268 (2.94 MP, 3757 tokens) fails and poisons the server;
/// the same image at 1.77 MP and below transcribes correctly. This cap sits
/// below the lowest known-bad point with margin to spare, and legibility has
/// room: that image still transcribed correctly down to 0.55 MP.
///
/// A cap on the long edge cannot do this job — a 2322x1268 grab is 40% more
/// area than a 2318x902 one at the same width.
const MAX_PIXELS: u32 = 1_750_000;

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
/// Scales `(w, h)` down until its area fits `MAX_PIXELS`, preserving the
/// aspect ratio. Images already within the cap are returned untouched, and
/// nothing is ever upscaled.
///
/// Area is the right invariant because image tokens scale with pixel count.
/// An earlier version capped the long edge and enforced a minimum short edge,
/// which let a 2322x1268 screenshot through at 2.94 MP — enough to make the
/// model confabulate and to poison the server for every later request. See
/// `MAX_PIXELS`.
#[must_use]
fn target_size(w: u32, h: u32) -> (u32, u32) {
    let area = u64::from(w) * u64::from(h);
    if area <= u64::from(MAX_PIXELS) || w == 0 || h == 0 {
        return (w, h);
    }
    // Both edges shrink by sqrt of the area ratio, so the shape is preserved.
    #[allow(clippy::cast_precision_loss)]
    let scale = (f64::from(MAX_PIXELS) / area as f64).sqrt();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let scaled = |v: u32| ((f64::from(v) * scale).floor() as u32).max(1);
    (scaled(w), scaled(h))
}

/// Composites an image carrying an alpha channel onto a white background,
/// returning an opaque image. These OCR models are trained overwhelmingly on
/// dark-on-light documents, so white is the only safe ground: a transparent
/// screenshot (window shadows, rounded corners, a checkbox rendered on a
/// glass surface) otherwise reaches the model as near-black RGB with the
/// real content hidden in the alpha channel, and the model confabulates a
/// confident, wrong transcription instead of failing loudly.
fn flatten_alpha(img: &image::DynamicImage) -> image::DynamicImage {
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    let mut out = image::RgbImage::new(width, height);
    for (x, y, px) in rgba.enumerate_pixels() {
        let [red, green, blue, alpha] = px.0;
        let alpha_f = f32::from(alpha) / 255.0;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let blend = |channel: u8| -> u8 {
            (f32::from(channel) * alpha_f + 255.0 * (1.0 - alpha_f)).round() as u8
        };
        out.put_pixel(x, y, image::Rgb([blend(red), blend(green), blend(blue)]));
    }
    image::DynamicImage::ImageRgb8(out)
}

/// Flattens alpha and downscales when needed, re-encoding as PNG; returns
/// the input unchanged if anything about the decode or re-encode fails,
/// since an oversized or mistranscribable image is better than no image.
///
/// Opaque images that need no resize pass through byte-for-byte: re-encoding
/// them would cost quality and time for no benefit.
fn normalize(media_type: &'static str, bytes: Vec<u8>) -> DecodedImage {
    let Ok(img) = image::load_from_memory(&bytes) else {
        return DecodedImage { media_type, bytes };
    };
    let has_alpha = img.color().has_alpha();
    let img = if has_alpha { flatten_alpha(&img) } else { img };

    let (tw, th) = target_size(img.width(), img.height());
    let needs_resize = (tw, th) != (img.width(), img.height());
    if !has_alpha && !needs_resize {
        return DecodedImage { media_type, bytes };
    }

    let img = if needs_resize {
        img.thumbnail(tw, th)
    } else {
        img
    };
    let mut out = std::io::Cursor::new(Vec::new());
    if img.write_to(&mut out, image::ImageFormat::Png).is_ok() {
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

    /// A 1x1 opaque RGB (no alpha channel) PNG, the smallest real image to
    /// exercise the decode paths.
    fn tiny_png() -> Vec<u8> {
        base64::engine::general_purpose::STANDARD
            .decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAD0lEQVR4AQEEAPv/AP8AAAMBAQCNHeWCAAAAAElFTkSuQmCC",
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
    /// encoding. Area is the binding limit, because image tokens scale with
    /// pixel count and an over-token image makes the model confabulate and
    /// poisons the server for every later request.
    #[test]
    fn every_shape_is_capped_by_area() {
        for (w, h) in [
            (2322, 1268), // the measured failure: 2.94 MP, poisoned the server
            (1700, 9000), // long scrollback grab
            (6000, 4000), // a 24 MP photo
            (9000, 1700), // the same tall case, rotated
        ] {
            let (ow, oh) = target_size(w, h);
            let area = u64::from(ow) * u64::from(oh);
            assert!(
                area <= u64::from(MAX_PIXELS),
                "{w}x{h} -> {ow}x{oh} = {area}px exceeds the cap"
            );
            // Aspect ratio must survive: squashing text is worse than shrinking it.
            let before = f64::from(w) / f64::from(h);
            let after = f64::from(ow) / f64::from(oh);
            assert!(
                (before - after).abs() / before < 0.01,
                "{w}x{h} -> {ow}x{oh} distorted the aspect ratio"
            );
        }
    }

    /// Anything already within the cap is passed through untouched — no
    /// upscaling, and no needless re-encode.
    #[test]
    fn images_within_the_cap_are_untouched() {
        assert_eq!(target_size(800, 600), (800, 600));
        // The known-good measured case: 2318x902 = 2.09 MP.
        assert_eq!(target_size(1400, 1200), (1400, 1200));
    }

    /// A fully transparent PNG whose alpha channel carries a distinct
    /// checkerboard pattern while RGB is uniformly mid-gray. If flattening
    /// worked correctly the "text" (alpha) must survive as visible contrast
    /// against a white background, and no pixel may remain transparent.
    fn transparent_checker_png(w: u32, h: u32) -> Vec<u8> {
        let mut img = image::RgbaImage::new(w, h);
        for (x, y, px) in img.enumerate_pixels_mut() {
            let dark = (x / 2 + y / 2) % 2 == 0;
            let a = if dark { 255 } else { 0 };
            *px = image::Rgba([10, 10, 10, a]);
        }
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    #[test]
    fn transparent_png_is_flattened_onto_white() {
        let png = transparent_checker_png(8, 8);
        let decoded = normalize("image/png", png);
        let out_img = image::load_from_memory(&decoded.bytes).unwrap().to_rgba8();

        // No pixel may remain transparent: the result must be fully opaque.
        for px in out_img.pixels() {
            assert_eq!(px.0[3], 255, "pixel still has alpha < 255: {px:?}");
        }
        // The previously fully-transparent (alpha=0) cells must now read as
        // white, not black: compositing onto white, not onto a dark ground.
        let corner = out_img.get_pixel(2, 0); // (2,0): dark=false -> was transparent
        assert!(
            corner.0[0] > 200 && corner.0[1] > 200 && corner.0[2] > 200,
            "expected near-white background pixel, got {corner:?}"
        );
        // The opaque "text" cells must still read as dark.
        let text_px = out_img.get_pixel(0, 0); // (0,0): dark=true -> was opaque
        assert!(
            text_px.0[0] < 60,
            "expected dark text pixel to survive flattening, got {text_px:?}"
        );
    }

    #[test]
    fn opaque_png_passes_through_byte_for_byte_when_no_resize_needed() {
        let png = tiny_png();
        let decoded = normalize("image/png", png.clone());
        assert_eq!(decoded.bytes, png, "opaque image should not be re-encoded");
    }

    #[test]
    fn transparent_and_oversized_image_gets_flattened_and_downscaled() {
        let png = transparent_checker_png(4000, 3000);
        let decoded = normalize("image/png", png);
        let out_img = image::load_from_memory(&decoded.bytes).unwrap();
        let area = u64::from(out_img.width()) * u64::from(out_img.height());
        assert!(area <= u64::from(MAX_PIXELS), "not downscaled: {area}px");
        let rgba = out_img.to_rgba8();
        for px in rgba.pixels() {
            assert_eq!(px.0[3], 255, "downscaled image still has transparency");
        }
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
