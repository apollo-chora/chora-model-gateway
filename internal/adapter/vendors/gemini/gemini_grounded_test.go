package gemini_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A canned Vertex "Grounding with Google Search" response.
const groundedResponseJSON = `{
  "candidates": [{
    "content": { "role": "model", "parts": [{"text": "The Great Wall is not visible to the naked eye from low Earth orbit without aid."}] },
    "finishReason": "STOP",
    "groundingMetadata": {
      "webSearchQueries": ["great wall visible from space"],
      "searchEntryPoint": { "renderedContent": "<style>.c{}</style><div class=\"container\">chip</div>" },
      "groundingChunks": [
        { "web": { "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc", "title": "NASA Earth Observatory", "domain": "nasa.gov" } },
        { "web": { "uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/def", "title": "Scientific American", "domain": "scientificamerican.com" } }
      ],
      "groundingSupports": [
        { "segment": { "startIndex": 0, "endIndex": 46, "text": "The Great Wall is not visible to the naked eye" }, "groundingChunkIndices": [0], "confidenceScores": [0.94] },
        { "segment": { "text": "from low Earth orbit without aid" }, "groundingChunkIndices": [1], "confidenceScores": [0.88] }
      ]
    }
  }],
  "usageMetadata": { "promptTokenCount": 30, "candidatesTokenCount": 90, "cachedContentTokenCount": 0 },
  "modelVersion": "gemini-2.5-flash-002"
}`

func TestGemini_GroundedGenerate_ParsesCitations(t *testing.T) {
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		assert.Contains(t, r.URL.Path, ":generateContent")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(groundedResponseJSON))
	}))
	defer srv.Close()

	c := newClient(t, srv)
	res, err := c.GroundedGenerate(context.Background(), domain.GroundedVendorRequest{
		LogicalModelID: "gemini-2.5-flash",
		Directive:      "Is the Great Wall of China visible from space?",
		MaxResults:     5,
		Traceparent:    "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
		TenantID:       "t",
	})
	require.NoError(t, err)

	// The request injected the google_search grounding tool.
	assert.Contains(t, string(gotBody), "google_search")

	// Answer + chip + queries.
	assert.Contains(t, res.Answer, "not visible to the naked eye")
	assert.Equal(t, "<style>.c{}</style><div class=\"container\">chip</div>", res.SearchEntryPointHTML)
	assert.Equal(t, []string{"great wall visible from space"}, res.WebSearchQueries)
	assert.Equal(t, domain.FinishReasonComplete, res.FinishReason)
	assert.Equal(t, "gemini-2.5-flash-002", res.ModelVersion)
	assert.Equal(t, int64(30), res.Usage.InputTokens)
	assert.Equal(t, int64(90), res.Usage.OutputTokens)

	// Two citations, domain rendered, snippet + confidence mapped from supports.
	require.Len(t, res.Citations, 2)
	assert.Equal(t, "nasa.gov", res.Citations[0].Domain)
	assert.Equal(t, "NASA Earth Observatory", res.Citations[0].Title)
	assert.Contains(t, res.Citations[0].URI, "grounding-api-redirect/abc")
	assert.Equal(t, "The Great Wall is not visible to the naked eye", res.Citations[0].Snippet)
	assert.InDelta(t, 0.94, res.Citations[0].Confidence, 0.001)
	assert.Equal(t, "scientificamerican.com", res.Citations[1].Domain)
	assert.Equal(t, "from low Earth orbit without aid", res.Citations[1].Snippet)
	assert.InDelta(t, 0.88, res.Citations[1].Confidence, 0.001)
}

func TestGemini_GroundedGenerate_ClampsAndRequestsGrounding(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		_, _ = w.Write([]byte(groundedResponseJSON))
	}))
	defer srv.Close()

	c := newClient(t, srv)
	_, err := c.GroundedGenerate(context.Background(), domain.GroundedVendorRequest{
		LogicalModelID: "gemini-2.5-flash",
		Directive:      "explain photosynthesis",
	})
	require.NoError(t, err)

	// tools carries a google_search entry.
	tools, ok := body["tools"].([]any)
	require.True(t, ok, "body.tools must be present")
	require.NotEmpty(t, tools)
	first, _ := tools[0].(map[string]any)
	_, hasGoogleSearch := first["google_search"]
	assert.True(t, hasGoogleSearch, "the grounding tool must be google_search")

	// The directive is the user content.
	contents, ok := body["contents"].([]any)
	require.True(t, ok)
	require.NotEmpty(t, contents)
}

func TestGemini_GroundedGenerate_ZeroGrounding_EmptyCitations(t *testing.T) {
	// A model reply with NO groundingMetadata ⇒ no citations (the caller hedges).
	const noGround = `{
      "candidates": [{ "content": { "role": "model", "parts": [{"text": "hi"}] }, "finishReason": "STOP" }],
      "usageMetadata": { "promptTokenCount": 5, "candidatesTokenCount": 2 },
      "modelVersion": "gemini-2.5-flash"
    }`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(noGround))
	}))
	defer srv.Close()

	c := newClient(t, srv)
	res, err := c.GroundedGenerate(context.Background(), domain.GroundedVendorRequest{
		LogicalModelID: "gemini-2.5-flash",
		Directive:      "hello",
	})
	require.NoError(t, err)
	assert.Empty(t, res.Citations)
	assert.Empty(t, res.SearchEntryPointHTML)
	assert.Empty(t, res.WebSearchQueries)
}

func TestGemini_GroundedGenerate_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":"overloaded"}`))
	}))
	defer srv.Close()

	c := newClient(t, srv)
	_, err := c.GroundedGenerate(context.Background(), domain.GroundedVendorRequest{
		LogicalModelID: "gemini-2.5-flash",
		Directive:      "x",
	})
	require.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), "503") || strings.Contains(err.Error(), "overloaded"))
}
