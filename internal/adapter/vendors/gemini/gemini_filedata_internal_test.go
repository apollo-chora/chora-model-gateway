package gemini

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
)

// TestBuildGeminiBody_FileDataRoundTrips asserts a fileData (gs:// by-reference)
// part in contents_json survives unmarshal → re-marshal into the Vertex body —
// the EPIC-1a batch-grounding multimodal carrier. A dropped fileData would mean
// the model never sees the uploaded source material.
func TestBuildGeminiBody_FileDataRoundTrips(t *testing.T) {
	contentsJSON := `[{"role":"user","parts":[` +
		`{"text":"Generate MCQs from the attached source."},` +
		`{"fileData":{"mimeType":"application/pdf","fileUri":"gs://chora-batch-uploads/t/job/material.pdf"}}` +
		`]}]`
	body := buildGeminiBody(domain.VendorRequest{ContentsJSON: contentsJSON, Prompt: "fallback"})

	if len(body.Contents) != 1 {
		t.Fatalf("Contents len = %d; want 1 (not the flat-prompt fallback)", len(body.Contents))
	}
	parts := body.Contents[0].Parts
	if len(parts) != 2 {
		t.Fatalf("parts len = %d; want 2 (text + fileData)", len(parts))
	}
	fd := parts[1].FileData
	if fd == nil {
		t.Fatal("fileData part dropped by buildGeminiBody")
	}
	if fd.FileURI != "gs://chora-batch-uploads/t/job/material.pdf" {
		t.Errorf("fileUri = %q", fd.FileURI)
	}
	if fd.MimeType != "application/pdf" {
		t.Errorf("mimeType = %q", fd.MimeType)
	}

	// And it must re-marshal to the Vertex `fileData` wire shape (the body the
	// adapter POSTs to :generateContent).
	raw, err := json.Marshal(body.Contents)
	if err != nil {
		t.Fatalf("marshal body.Contents: %v", err)
	}
	if !strings.Contains(string(raw), `"fileData"`) || !strings.Contains(string(raw), `"fileUri"`) {
		t.Errorf("re-marshaled body missing fileData/fileUri: %s", raw)
	}
}
