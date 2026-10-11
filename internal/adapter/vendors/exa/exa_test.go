package exa_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/vendors/exa"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// stubServers pairs an Exa search stub with a LongCat chat stub, capturing
// what each received.
type stubServers struct {
	exa       *httptest.Server
	chat      *httptest.Server
	exaCalls  atomic.Int32
	chatCalls atomic.Int32

	lastExaBody  map[string]any
	lastChatBody map[string]any
}

func newStubServers(t *testing.T, exaStatus int, exaBody string, chatStatus int, chatBody string) *stubServers {
	t.Helper()
	s := &stubServers{}
	s.exa = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.exaCalls.Add(1)
		if r.Header.Get("X-API-KEY") != "exa-key" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		_ = json.NewDecoder(r.Body).Decode(&s.lastExaBody)
		w.WriteHeader(exaStatus)
		_, _ = w.Write([]byte(exaBody))
	}))
	s.chat = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.chatCalls.Add(1)
		_ = json.NewDecoder(r.Body).Decode(&s.lastChatBody)
		w.WriteHeader(chatStatus)
		_, _ = w.Write([]byte(chatBody))
	}))
	t.Cleanup(func() { s.exa.Close(); s.chat.Close() })
	return s
}

func newClient(s *stubServers) *exa.Client {
	c, err := exa.New(exa.Config{
		HTTPClient:   s.exa.Client(),
		APIKey:       "exa-key",
		ChatEndpoint: s.chat.URL,
		ChatAPIKey:   "chat-key",
		ExaHost:      s.exa.URL,
	})
	if err != nil {
		panic(err)
	}
	return c
}

// Compile-time check that the exa Client implements the grounded vendor port
// (ADR-231 Exa cutover 2026-10-10 — previously the gemini adapter).
var _ domain.GroundedVendorClient = (*exa.Client)(nil)

func TestExa_Family(t *testing.T) {
	s := newStubServers(t, 200, `{"results":[]}`, 200, `{"choices":[]}`)
	assert.Equal(t, domain.VendorFamilyExa, newClient(s).Family())
}

func TestExa_GroundedGenerate_HappyPath(t *testing.T) {
	s := newStubServers(t, 200, `{"results":[
		{"url":"https://a.example.com/x","title":"A","text":"alpha body","score":0.9},
		{"url":"https://b.example.com/y","title":"B","text":"beta body","score":0.8}
	]}`, 200, `{"choices":[{"message":{"content":"synthesised"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":10}}`)

	resp, err := newClient(s).GroundedGenerate(context.Background(), domain.GroundedVendorRequest{
		LogicalModelID: "longcat-2.5-preview",
		Directive:      "what is alpha",
		MaxResults:     5,
	})
	require.NoError(t, err)

	assert.Equal(t, "synthesised", resp.Answer)
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
	assert.Equal(t, "LongCat-2.5-Preview", resp.ModelVersion)
	assert.Equal(t, []string{"what is alpha"}, resp.WebSearchQueries)
	assert.Empty(t, resp.SearchEntryPointHTML, "Exa has no Google search-entry chip")

	require.Len(t, resp.Citations, 2)
	assert.Equal(t, "https://a.example.com/x", resp.Citations[0].URI)
	assert.Equal(t, "A", resp.Citations[0].Title)
	assert.Equal(t, "a.example.com", resp.Citations[0].Domain)
	assert.Equal(t, "alpha body", resp.Citations[0].Snippet)
	assert.InDelta(t, 0.9, resp.Citations[0].Confidence, 0.001)

	// Usage = Exa surcharge + LongCat token cost (100 in, 10 out).
	assert.Equal(t, int64(100), resp.Usage.InputTokens)
	assert.Equal(t, int64(10), resp.Usage.OutputTokens)
	assert.Equal(t, int64(1), resp.Usage.GroundingUnits)
	assert.Equal(t, int64(10_000+100*300_000/1_000_000+10*1_200_000/1_000_000), resp.Usage.CostMicros)

	// The directive reached Exa; the sources were fenced as untrusted data.
	assert.Equal(t, "what is alpha", s.lastExaBody["query"])
	_, hasContents := s.lastExaBody["contents"]
	assert.True(t, hasContents, "Exa request must ask for page text")
	user := s.lastChatBody["messages"].([]any)[1].(map[string]any)["content"].(string)
	assert.Contains(t, user, "UNTRUSTED CONTENT")
	assert.Contains(t, user, "alpha body")
}

func TestExa_GroundedGenerate_ZeroResults_InBandEmpty(t *testing.T) {
	s := newStubServers(t, 200, `{"results":[]}`, 200, `{"choices":[]}`)

	resp, err := newClient(s).GroundedGenerate(context.Background(), domain.GroundedVendorRequest{
		Directive: "nothing findable",
	})
	require.NoError(t, err, "zero citations is an in-band anomaly, not an error")
	assert.Empty(t, resp.Citations)
	assert.Empty(t, resp.Answer)
	assert.Equal(t, int64(10_000), resp.Usage.CostMicros, "the Exa search still billed")
	assert.Equal(t, int32(0), s.chatCalls.Load(), "no synthesis without evidence")
}

func TestExa_GroundedGenerate_DedupesAndCaps(t *testing.T) {
	s := newStubServers(t, 200, `{"results":[
		{"url":"https://a.example.com/1","title":"A1","text":"one"},
		{"url":"https://a.example.com/1","title":"A1-dup","text":"one-again"},
		{"url":"https://a.example.com/2","title":"A2","text":"two"},
		{"url":"https://a.example.com/3","title":"A3","text":"three"},
		{"url":"","title":"no-url","text":"drop"}
	]}`, 200, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}`)

	resp, err := newClient(s).GroundedGenerate(context.Background(), domain.GroundedVendorRequest{
		Directive:  "q",
		MaxResults: 2,
	})
	require.NoError(t, err)
	require.Len(t, resp.Citations, 2, "deduped by URL, capped at maxResults, url-less dropped")
	assert.Equal(t, "A1", resp.Citations[0].Title)
	assert.Equal(t, "A2", resp.Citations[1].Title)
}

func TestExa_GroundedGenerate_RateLimitRetriesThenSucceeds(t *testing.T) {
	var calls atomic.Int32
	s := &stubServers{}
	s.exa = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = w.Write([]byte(`{"results":[{"url":"https://a.example.com/x","title":"A","text":"alpha"}]}`))
	}))
	s.chat = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}`))
	}))
	t.Cleanup(func() { s.exa.Close(); s.chat.Close() })

	resp, err := newClient(s).GroundedGenerate(context.Background(), domain.GroundedVendorRequest{Directive: "q"})
	require.NoError(t, err)
	assert.Len(t, resp.Citations, 1)
	assert.Equal(t, int32(2), calls.Load(), "one 429 then a retry")
}

func TestExa_GroundedGenerate_AuthFailureFailsClosed(t *testing.T) {
	s := newStubServers(t, 403, `{"error":"bad key"}`, 200, `{"choices":[]}`)

	_, err := newClient(s).GroundedGenerate(context.Background(), domain.GroundedVendorRequest{Directive: "q"})
	require.Error(t, err)
	var perr *domain.ProviderError
	require.True(t, errors.As(err, &perr))
	assert.Equal(t, 403, perr.Status)
	assert.Equal(t, int32(1), s.exaCalls.Load(), "403 is not retried")
	assert.Equal(t, int32(0), s.chatCalls.Load())
}

func TestExa_GroundedGenerate_SynthesisFailureCarriesPartialUsage(t *testing.T) {
	s := newStubServers(t, 200, `{"results":[{"url":"https://a.example.com/x","title":"A","text":"alpha"}]}`, 500, `{"error":"boom"}`)

	resp, err := newClient(s).GroundedGenerate(context.Background(), domain.GroundedVendorRequest{Directive: "q"})
	require.Error(t, err, "synthesis failure is an error")
	assert.Equal(t, int64(10_000), resp.Usage.CostMicros, "the Exa search billed even though synthesis failed")
	assert.Equal(t, int64(1), resp.Usage.GroundingUnits)
	assert.Equal(t, int32(4), s.chatCalls.Load(), "initial + 3 retries")
}

func TestExa_GroundedGenerate_MissingAPIKeyFailsClosed(t *testing.T) {
	c, err := exa.New(exa.Config{})
	require.NoError(t, err)
	_, err = c.GroundedGenerate(context.Background(), domain.GroundedVendorRequest{Directive: "q"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "EXA_API_KEY")
}
