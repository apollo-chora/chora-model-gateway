package domain

import (
	"encoding/json"
	"strings"
)

// MaxScreenableContentsBytes bounds the text derived from `contents_json`
// before it is handed to Cloud Model Armor.
//
// Two reasons for a bound rather than sending everything. Armor's sanitize API
// has a request-size limit, and a learner artifact is arbitrarily large (a
// scanned test transcribed to text runs to tens of kilobytes), so an unbounded
// concatenation turns a large-but-legitimate upload into a screening FAILURE,
// which on an audit-only leg reads as "unscreened" and on an enforcing leg
// would refuse the upload outright. Truncation degrades detection at the tail;
// an unbounded send degrades it everywhere.
const MaxScreenableContentsBytes = 8192

// DeriveScreenableText extracts the model-facing TEXT from an ADR-177
// `contents_json` payload so Armor PRE can screen what the model will actually
// read, rather than whatever the caller happened to put in the flat `prompt`.
//
// WHY THIS EXISTS (G1'-3). chora-ai-kernel-orchestrator's multimodal client
// sends a fixed placeholder as `prompt` and puts the learner's uploaded
// artifact in `contents_json`. The gateway screens `prompt`, so the constant is
// screened and the artifact is not. The Go client
// (poc/chora-adk-common/modelgatewayclient/client.go) does the right thing and
// always sends real text in `prompt`; the Python client diverges from that
// contract. Deriving the text HERE means no caller can opt out of screening by
// getting the contract wrong, which is the whole point of a chokepoint.
//
// WHAT IT DELIBERATELY DOES NOT DO. `inlineData` parts (base64 image or PDF
// bytes) are skipped. They are not text, Cloud Model Armor is a text guardrail,
// and feeding base64 to it produces noise rather than safety. Screening the
// bytes themselves needs an image-safety classifier and is tracked separately
// as G1'-2.
//
// Returns "" for absent, malformed or text-free contents. A parse failure is
// NOT an error: this runs on the chokepoint's hot path, and a caller sending a
// shape we do not recognise must not take their turn down. The empty return is
// the signal that there was nothing to screen, and the caller decides what that
// means.
func DeriveScreenableText(contentsJSON string) string {
	s := strings.TrimSpace(contentsJSON)
	if s == "" {
		return ""
	}

	var turns []struct {
		Parts json.RawMessage `json:"parts"`
	}
	if err := json.Unmarshal([]byte(s), &turns); err != nil {
		return ""
	}

	var b strings.Builder
	for _, turn := range turns {
		if len(turn.Parts) == 0 {
			continue
		}
		// `parts` is a list in the genai shape. Anything else is a caller bug,
		// and is skipped rather than guessed at.
		var parts []struct {
			Text *string `json:"text"`
		}
		if err := json.Unmarshal(turn.Parts, &parts); err != nil {
			continue
		}
		for _, p := range parts {
			// A non-string `text` unmarshals to nil here, so a malformed part
			// contributes nothing instead of contributing "0" or "<nil>".
			if p.Text == nil || *p.Text == "" {
				continue
			}
			if b.Len() > 0 {
				b.WriteByte('\n')
			}
			b.WriteString(*p.Text)
			if b.Len() >= MaxScreenableContentsBytes {
				return truncateUTF8(b.String(), MaxScreenableContentsBytes)
			}
		}
	}
	return truncateUTF8(b.String(), MaxScreenableContentsBytes)
}

// truncateUTF8 cuts s to at most n BYTES without splitting a rune, so the
// screened text is always valid UTF-8. Armor rejects invalid UTF-8, and a
// rejection on this leg would read as a screening outage rather than as a
// truncation.
func truncateUTF8(s string, n int) string {
	if len(s) <= n {
		return s
	}
	cut := n
	for cut > 0 && !isUTF8Start(s[cut]) {
		cut--
	}
	return s[:cut]
}

// isUTF8Start reports whether b begins a UTF-8 rune (i.e. is not a 10xxxxxx
// continuation byte).
func isUTF8Start(b byte) bool { return b&0xC0 != 0x80 }
