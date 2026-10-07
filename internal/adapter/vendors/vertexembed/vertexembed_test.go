package vertexembed

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
)

type staticTokens struct{ tok string }

func (s staticTokens) Token(context.Context) (string, error) { return s.tok, nil }

func TestNew_requiresConfig(t *testing.T) {
	if _, err := New(Config{Project: "p", Location: "l"}); err == nil {
		t.Fatal("nil TokenProvider must be refused")
	}
	if _, err := New(Config{Tokens: staticTokens{"t"}, Location: "l"}); err == nil {
		t.Fatal("empty Project must be refused")
	}
	if _, err := New(Config{Tokens: staticTokens{"t"}, Project: "p"}); err == nil {
		t.Fatal("empty Location must be refused")
	}
}

func TestEmbedText_happyPath(t *testing.T) {
	var gotPath, gotAuth string
	var gotBody predictRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"predictions":[{"embeddings":{"statistics":{"token_count":9},"values":[0.5,0.25]}}]}`))
	}))
	defer srv.Close()

	c, err := New(Config{Tokens: staticTokens{"tok-1"}, Project: "chora-489812", Location: "asia-southeast1", Endpoint: srv.URL, HTTPClient: srv.Client()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := c.EmbedText(context.Background(), domain.EmbedVendorRequest{
		LogicalModelID:   "text-embedding-004",
		Text:             "fractions",
		TaskType:         "RETRIEVAL_QUERY",
		OutputDimensions: 768,
	})
	if err != nil {
		t.Fatalf("EmbedText: %v", err)
	}
	if len(resp.Values) != 2 || resp.Values[0] != 0.5 {
		t.Errorf("values = %v", resp.Values)
	}
	if resp.InputTokens != 9 {
		t.Errorf("vendor token_count must surface; got %d", resp.InputTokens)
	}
	if resp.ModelVersion != "text-embedding-004" {
		t.Errorf("model_version = %q", resp.ModelVersion)
	}
	if !strings.Contains(gotPath, "publishers/google/models/text-embedding-004:predict") {
		t.Errorf("path = %q", gotPath)
	}
	if gotAuth != "Bearer tok-1" {
		t.Errorf("auth = %q", gotAuth)
	}
	if gotBody.Parameters.OutputDimensionality != 768 || gotBody.Instances[0].TaskType != "RETRIEVAL_QUERY" {
		t.Errorf("body = %+v", gotBody)
	}
}

func TestEmbedText_defaultsTaskTypeToDocument(t *testing.T) {
	var gotBody predictRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_, _ = w.Write([]byte(`{"predictions":[{"embeddings":{"values":[0.1]}}]}`))
	}))
	defer srv.Close()
	c, _ := New(Config{Tokens: staticTokens{"t"}, Project: "p", Location: "l", Endpoint: srv.URL, HTTPClient: srv.Client()})
	if _, err := c.EmbedText(context.Background(), domain.EmbedVendorRequest{LogicalModelID: "text-embedding-004", Text: "x", OutputDimensions: 768}); err != nil {
		t.Fatalf("EmbedText: %v", err)
	}
	if gotBody.Instances[0].TaskType != "RETRIEVAL_DOCUMENT" {
		t.Errorf("empty task type must default to RETRIEVAL_DOCUMENT; got %q", gotBody.Instances[0].TaskType)
	}
}

func TestEmbedText_httpErrorSurfacesLoud(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"quota"}`, http.StatusTooManyRequests)
	}))
	defer srv.Close()
	c, _ := New(Config{Tokens: staticTokens{"t"}, Project: "p", Location: "l", Endpoint: srv.URL, HTTPClient: srv.Client()})
	_, err := c.EmbedText(context.Background(), domain.EmbedVendorRequest{LogicalModelID: "text-embedding-004", Text: "x", OutputDimensions: 768})
	if err == nil || !strings.Contains(err.Error(), "429") {
		t.Fatalf("HTTP error must surface with status; got %v", err)
	}
}

func TestEmbedText_emptyPredictionIsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"predictions":[]}`))
	}))
	defer srv.Close()
	c, _ := New(Config{Tokens: staticTokens{"t"}, Project: "p", Location: "l", Endpoint: srv.URL, HTTPClient: srv.Client()})
	if _, err := c.EmbedText(context.Background(), domain.EmbedVendorRequest{LogicalModelID: "text-embedding-004", Text: "x", OutputDimensions: 768}); err == nil {
		t.Fatal("empty prediction must be an error, never a silent zero-vector")
	}
}
