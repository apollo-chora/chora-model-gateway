package openai

import (
	"context"
	"encoding/base64"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// TestSniffImageMIME pins the magic-byte table. The WebP and JPEG cases are
// here because api.meta.ai's muse-image-1.0 returns WebP (observed live
// 2026-10-04) while the OpenAI docs promise PNG — the whole point of sniffing
// is that those disagree.
func TestSniffImageMIME(t *testing.T) {
	cases := []struct {
		name string
		in   []byte
		want string
	}{
		{"png", append([]byte("\x89PNG\r\n\x1a\n"), make([]byte, 20)...), mimePNG},
		{"jpeg", append([]byte{0xFF, 0xD8, 0xFF, 0xE0}, make([]byte, 20)...), mimeJPEG},
		{"webp riff", append([]byte("RIFF\x24\x00\x00\x00WEBPVP8 "), make([]byte, 20)...), mimeWebP},
		{"gif87", append([]byte("GIF87a"), make([]byte, 20)...), mimeGIF},
		{"gif89", append([]byte("GIF89a"), make([]byte, 20)...), mimeGIF},
		{"empty", nil, mimePNG},
		{"unknown bytes", []byte("not an image at all"), mimePNG},
		// A RIFF container that is NOT WebP (WAV is also RIFF) must not be
		// misreported as an image.
		{"riff wave", append([]byte("RIFF\x24\x00\x00\x00WAVEfmt "), make([]byte, 20)...), mimePNG},
		// Too short to identify.
		{"truncated png header", []byte("\x89PN"), mimePNG},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sniffImageMIME(tc.in); got != tc.want {
				t.Errorf("sniffImageMIME = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestGenerate_ImageMIMEIsSniffedNotAssumed is the end-to-end guard: a provider
// that returns WebP must not be reported as PNG.
func TestGenerate_ImageMIMEIsSniffedNotAssumed(t *testing.T) {
	c, up := newTestClient(t, "")
	webp := append([]byte("RIFF\x24\x00\x00\x00WEBPVP8 "), make([]byte, 16)...)
	up.respondJSON(`{"data":[{"b64_json":"` +
		base64EncodeForTest(webp) + `"}]}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:           target(up.URL),
		Prompt:           "a cube",
		ResponseModality: domain.ModalityImage,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.ImageMIMEType != mimeWebP {
		t.Errorf("mime = %q, want %q — a WebP labelled as a PNG will not render",
			resp.ImageMIMEType, mimeWebP)
	}
}

func base64EncodeForTest(b []byte) string {
	return base64.StdEncoding.EncodeToString(b)
}
