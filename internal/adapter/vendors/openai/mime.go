package openai

import "bytes"

// Image formats a provider can hand back inside data[0].b64_json.
//
// This exists because the mime type CANNOT be assumed. OpenAI's images endpoint
// documents PNG, but the OpenAI-compatible aggregators do not all honour that:
// api.meta.ai's muse-image-1.0 returns a WebP, verified live 2026-10-04 with
// the `RIFF....WEBP` magic. Hardcoding "image/png" on such a response makes the
// gateway serve a WebP labelled as a PNG, which a browser will refuse to
// render — the bytes decode, the Content-Type lies, and the failure surfaces far
// from the cause.
//
// Sniffing the magic bytes is the only reliable signal here: the provider sends
// no Content-Type for an inline image, and the `response_format` a caller asked
// for describes what the caller WANTS, not what came back.
const (
	mimePNG  = "image/png"
	mimeJPEG = "image/jpeg"
	mimeWebP = "image/webp"
	mimeGIF  = "image/gif"
)

// sniffImageMIME identifies an image from its leading bytes, defaulting to PNG
// when nothing matches.
//
// The PNG default is deliberate: it is what OpenAI documents and what the
// overwhelming majority of providers send, so an unrecognised format is far more
// likely to be an unusual PNG than an unusual PNG-alike.
func sniffImageMIME(b []byte) string {
	switch {
	case len(b) >= 8 && bytes.Equal(b[:8], []byte("\x89PNG\r\n\x1a\n")):
		return mimePNG
	case len(b) >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF:
		return mimeJPEG
	case len(b) >= 12 && bytes.Equal(b[:4], []byte("RIFF")) && bytes.Equal(b[8:12], []byte("WEBP")):
		return mimeWebP
	case len(b) >= 6 && (bytes.Equal(b[:6], []byte("GIF87a")) || bytes.Equal(b[:6], []byte("GIF89a"))):
		return mimeGIF
	default:
		return mimePNG
	}
}
