package gemini_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/adapter/vendors/gemini"
	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type stubTokens struct {
	token string
	err   error
}

func (s stubTokens) Token(ctx context.Context) (string, error) { return s.token, s.err }

func newClient(t *testing.T, srv *httptest.Server) *gemini.Client {
	t.Helper()
	c, err := gemini.New(gemini.Config{
		HTTPClient: srv.Client(),
		Tokens:     stubTokens{token: "test-bearer"},
		Project:    "chora-489812",
		Location:   "asia-southeast1",
		Endpoint:   srv.URL,
	})
	require.NoError(t, err)
	return c
}

func TestGemini_Family(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()
	c := newClient(t, srv)
	assert.Equal(t, domain.VendorFamilyVertexGemini, c.Family())
}

func TestGemini_GenerateHappyPath(t *testing.T) {
	var capturedAuth, capturedPath, capturedTraceparent string
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")
		capturedPath = r.URL.Path
		capturedTraceparent = r.Header.Get("traceparent")
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"candidates": [{
				"content": {"role":"model","parts":[{"text":"two-week Sprint Planning is timeboxed at 4 hours"}]},
				"finishReason": "STOP"
			}],
			"usageMetadata": {
				"promptTokenCount": 100,
				"candidatesTokenCount": 50,
				"cachedContentTokenCount": 10
			},
			"modelVersion": "gemini-2.5-pro@001"
		}`))
	}))
	defer srv.Close()

	c := newClient(t, srv)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Vendor:           domain.VendorFamilyVertexGemini,
		LogicalModelID:   domain.LogicalModelID("gemini-2.5-pro"),
		Prompt:           "Scrum cadence?",
		SystemPrompt:     "You are an expert curriculum author.",
		GenerationConfig: map[string]any{"temperature": 0.7},
		Traceparent:      "00-trace-id-span-id-01",
	})
	require.NoError(t, err)

	// Request shape — Authorization + path + traceparent + body.
	assert.Equal(t, "Bearer test-bearer", capturedAuth)
	assert.Contains(t, capturedPath, "/v1/projects/chora-489812/locations/asia-southeast1/publishers/google/models/gemini-2.5-pro:generateContent")
	assert.Equal(t, "00-trace-id-span-id-01", capturedTraceparent)
	require.NotNil(t, capturedBody["contents"])
	require.NotNil(t, capturedBody["systemInstruction"])
	require.NotNil(t, capturedBody["generationConfig"])

	// Response shape — completion, model version, usage + cost computed.
	assert.Equal(t, "two-week Sprint Planning is timeboxed at 4 hours", resp.Completion)
	assert.Equal(t, "gemini-2.5-pro@001", resp.ModelVersion)
	assert.Equal(t, int64(100), resp.Usage.InputTokens)
	assert.Equal(t, int64(50), resp.Usage.OutputTokens)
	assert.Equal(t, int64(10), resp.Usage.CachedTokens)
	// Pricing is micro-USD PER 1M tokens (gemini-2.5-pro: 1_250_000 input +
	// 10_000_000 output + 130_000 cached — $1.25 / $10.00 / $0.13 per 1M).
	// Fresh input = 100 - 10 cached = 90 tokens.
	// Cost = 90 * 1_250_000 / 1e6   =  112 (truncated from 112.5)
	//      + 10 * 130_000   / 1e6   =    1 (truncated from 1.3)
	//      + 50 * 10_000_000 / 1e6  =  500
	//                               =  613 micro-USD.
	//
	// CHO-2220: asserted 362 originally, from a $5.00/1M output rate — half the
	// real price. Then 612, when cached tokens were still billed at zero. Both
	// assertions were faithful to the table; the table was wrong both times, so
	// the test stayed green while the ledger under-billed. Re-derive from the
	// published price list, never from whatever the code currently returns.
	assert.Equal(t, int64(613), resp.Usage.CostMicros)
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
}

// tinyPNG is a 1x1 transparent PNG (the smallest valid PNG) used to assert
// the adapter base64-decodes inlineData into VendorResponse.ImageBytes byte-
// for-byte. The constant is the raw bytes; the test base64-encodes it into the
// fake vendor response just as Vertex AI returns it.
var tinyPNG = []byte{
	0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
	0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
	0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00,
	0x0a, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x00, 0x01, 0x00, 0x00,
	0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
	0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
}

// TestGemini_GenerateImageResponse asserts the adapter base64-decodes an
// inlineData candidate part into VendorResponse.ImageBytes (byte-exact) +
// surfaces the MIME type. An image-only response carries no text part — the
// W8 image-modality contract (CR 2026-06-01).
func TestGemini_GenerateImageResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Vertex returns the PNG as standard base64 inside inlineData.data.
		// Image-only candidate: a single inlineData part, no text part.
		_, _ = fmt.Fprintf(w, `{
			"candidates": [{
				"content": {"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":%q}}]},
				"finishReason": "STOP"
			}],
			"usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 0},
			"modelVersion": "gemini-3-pro-image@001"
		}`, base64.StdEncoding.EncodeToString(tinyPNG))
	}))
	defer srv.Close()
	c := newClient(t, srv)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Vendor:           domain.VendorFamilyVertexGemini,
		LogicalModelID:   domain.LogicalModelID("gemini-3-pro-image"),
		Prompt:           "draw a red square",
		ResponseModality: "IMAGE",
	})
	require.NoError(t, err)

	// Image bytes decode byte-for-byte; MIME type surfaced; no text completion.
	assert.Equal(t, tinyPNG, resp.ImageBytes)
	assert.Equal(t, "image/png", resp.ImageMIMEType)
	assert.Equal(t, "", resp.Completion)
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
}

// TestGemini_GenerateImageRequestSetsResponseModalities asserts that an
// IMAGE-modality VendorRequest places responseModalities:["IMAGE"] under
// generationConfig in the outbound :generateContent body (the documented
// Vertex REST location).
func TestGemini_GenerateImageRequestSetsResponseModalities(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{
			"candidates": [{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":%q}}]},"finishReason":"STOP"}],
			"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 0}
		}`, base64.StdEncoding.EncodeToString(tinyPNG))
	}))
	defer srv.Close()
	c := newClient(t, srv)

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		LogicalModelID:   domain.LogicalModelID("gemini-3-pro-image"),
		Prompt:           "draw a red square",
		ResponseModality: "IMAGE",
	})
	require.NoError(t, err)

	// generationConfig.responseModalities == ["IMAGE"]. JSON decodes arrays
	// to []any and strings to string, so assert on that shape.
	require.NotNil(t, capturedBody["generationConfig"], "generationConfig must be set for IMAGE requests")
	genCfg, ok := capturedBody["generationConfig"].(map[string]any)
	require.True(t, ok, "generationConfig must be an object")
	mods, ok := genCfg["responseModalities"].([]any)
	require.True(t, ok, "responseModalities must be an array under generationConfig")
	require.Len(t, mods, 1)
	assert.Equal(t, "IMAGE", mods[0])
}

// TestGemini_GenerateTextRequestOmitsResponseModalities asserts a plain
// (TEXT/empty) request does NOT inject responseModalities — the field is
// omitempty so a text-only body stays byte-identical to pre-W8 behaviour.
func TestGemini_GenerateTextRequestOmitsResponseModalities(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}
		}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		LogicalModelID: domain.LogicalModelID("gemini-2.5-pro"),
		Prompt:         "hello",
		// ResponseModality intentionally empty (TEXT).
	})
	require.NoError(t, err)

	// No generationConfig was supplied + no modality → field absent entirely.
	if genCfg, ok := capturedBody["generationConfig"].(map[string]any); ok {
		_, present := genCfg["responseModalities"]
		assert.False(t, present, "responseModalities must be absent for TEXT requests")
	}
}

// TestGemini_GenerateImageAndText asserts a candidate carrying BOTH a text
// part and an inlineData part surfaces both (text completion + image bytes).
func TestGemini_GenerateImageAndText(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{
			"candidates": [{
				"content": {"role":"model","parts":[
					{"text":"here is your square:"},
					{"inlineData":{"mimeType":"image/png","data":%q}}
				]},
				"finishReason": "STOP"
			}],
			"usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4}
		}`, base64.StdEncoding.EncodeToString(tinyPNG))
	}))
	defer srv.Close()
	c := newClient(t, srv)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		LogicalModelID:   domain.LogicalModelID("gemini-3-pro-image"),
		Prompt:           "draw a red square and describe it",
		ResponseModality: "IMAGE",
	})
	require.NoError(t, err)
	assert.Equal(t, "here is your square:", resp.Completion)
	assert.Equal(t, tinyPNG, resp.ImageBytes)
	assert.Equal(t, "image/png", resp.ImageMIMEType)
}

// TestGemini_GenerateImageBadBase64 asserts a malformed inlineData.data fails
// loud (per directive) rather than silently returning empty image bytes.
func TestGemini_GenerateImageBadBase64(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"!!!not-base64!!!"}}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":0}
		}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	_, err := c.Generate(context.Background(), domain.VendorRequest{
		LogicalModelID:   domain.LogicalModelID("gemini-3-pro-image"),
		Prompt:           "draw",
		ResponseModality: "IMAGE",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "inlineData")
}

func TestGemini_GenerateSafetyBlock(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"candidates": [{
				"content": {"role":"model","parts":[{"text":""}]},
				"finishReason": "SAFETY"
			}],
			"usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 0}
		}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	resp, err := c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: domain.LogicalModelID("gemini-2.5-flash")})
	require.NoError(t, err)
	assert.Equal(t, domain.FinishReasonModelArmorBlock, resp.FinishReason)
}

func TestGemini_GenerateHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"backend overloaded"}`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	_, err := c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: domain.LogicalModelID("gemini-2.5-pro")})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "503")
	assert.Contains(t, err.Error(), "backend overloaded")
}

func TestGemini_GenerateTokenError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()
	c, err := gemini.New(gemini.Config{
		HTTPClient: srv.Client(),
		Tokens:     stubTokens{err: errors.New("WIF token mint failed")},
		Project:    "chora-489812",
		Location:   "asia-southeast1",
		Endpoint:   srv.URL,
	})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "gemini-2.5-pro"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "WIF token mint failed")
}

func TestGemini_GenerateUnknownModel_ZeroCost(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = fmt.Fprint(w, `{
			"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}
		}`)
	}))
	defer srv.Close()
	c := newClient(t, srv)
	resp, err := c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: domain.LogicalModelID("gemini-vapor-mlm-X")})
	require.NoError(t, err)
	assert.Equal(t, int64(0), resp.Usage.CostMicros)
}

func TestGemini_GenerateInvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`not json`))
	}))
	defer srv.Close()
	c := newClient(t, srv)
	_, err := c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "gemini-2.5-pro"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unmarshal")
}

func TestGemini_NewValidation(t *testing.T) {
	cases := []struct {
		name string
		cfg  gemini.Config
		want string
	}{
		{"missing tokens", gemini.Config{Project: "x", Location: "asia-southeast1"}, "TokenProvider"},
		{"missing project", gemini.Config{Tokens: stubTokens{}, Location: "asia-southeast1"}, "Project"},
		{"missing location", gemini.Config{Tokens: stubTokens{}, Project: "x"}, "Location"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := gemini.New(tc.cfg)
			require.Error(t, err)
			assert.True(t, strings.Contains(err.Error(), tc.want), "got %q", err.Error())
		})
	}
}

func TestGemini_NewDefaultEndpoint(t *testing.T) {
	// Bypasses the test endpoint override so we exercise the production
	// host-construction code path. The HTTP client is set to a stub that
	// rejects every call so the test is hermetic.
	hc := &http.Client{Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		// Confirm we built the right URL.
		assert.Equal(t, "asia-southeast1-aiplatform.googleapis.com", req.URL.Host)
		return nil, errors.New("hermetic-stop")
	})}
	c, err := gemini.New(gemini.Config{
		HTTPClient: hc,
		Tokens:     stubTokens{token: "t"},
		Project:    "chora-489812",
		Location:   "asia-southeast1",
	})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "gemini-2.5-pro"})
	require.Error(t, err)
}

func TestGemini_GlobalLocationHost(t *testing.T) {
	// location="global" routes to the un-prefixed global endpoint
	// (https://aiplatform.googleapis.com) with `locations/global` in the
	// path — Gemini 3.x text + image models are global-only (CR qgen
	// 2026-06-01). Hermetic: the transport asserts host + path then aborts.
	var capturedHost, capturedPath string
	hc := &http.Client{Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		capturedHost = req.URL.Host
		capturedPath = req.URL.Path
		return nil, errors.New("hermetic-stop")
	})}
	c, err := gemini.New(gemini.Config{
		HTTPClient: hc,
		Tokens:     stubTokens{token: "t"},
		Project:    "chora-489812",
		Location:   "global",
	})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "gemini-3.1-pro-preview"})
	require.Error(t, err)
	assert.Equal(t, "aiplatform.googleapis.com", capturedHost)
	assert.Contains(t, capturedPath, "/v1/projects/chora-489812/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent")
}

type roundTripperFunc func(req *http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }
