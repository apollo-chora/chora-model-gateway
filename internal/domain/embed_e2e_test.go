package domain_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	registryadapter "github.com/apollo-chora/chora-model-gateway/internal/adapter/registry"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/vendors/openai"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	modelregistry "github.com/apollo-chora/chora-model-gateway/internal/registry"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// End-to-end proof of the embedding dimension policy across every layer the
// production request crosses: the REAL YAML registry loader, the REAL
// registry→domain resolver adapter, the REAL domain Embed flow and the REAL
// OpenAI-compatible HTTP adapter, against an httptest endpoint that stands in
// for OpenRouter.
//
// The defect this pins down: the gateway used to send `dimensions: 768` to a
// model that produces 1024 values and rejects the parameter, so an omitted
// dimension (the normal caller) turned a working route into a 400. Here the
// caller omits `dimensions` entirely and the request must reach the wire with
// no dimensions field at all, while the 1024-element response is accepted.

// e2eSecretClient is the SecretClient the adapter is constructed with. It
// deliberately returns a DIFFERENT key for the vendor family than the entry
// carries: the request must use the resolved entry's credential.
type e2eSecretClient struct{ familyKey string }

func (s e2eSecretClient) ResolveByoaKey(context.Context, string, domain.VendorFamily) (string, error) {
	return s.familyKey, nil
}

func (s e2eSecretClient) ResolveGlobalKey(context.Context, domain.VendorFamily) (string, error) {
	return s.familyKey, nil
}

func writeE2ERegistry(t *testing.T, baseURL string) string {
	t.Helper()
	// The registry loader refuses a literal private/loopback IP (the SSRF
	// guard), so the test endpoint is addressed by name.
	baseURL = strings.Replace(baseURL, "127.0.0.1", "localhost", 1)
	content := `
models:
  - id: text-embedding-004
    provider: openai
    upstream_model: liquid/lfm-2.5-embedding-350m:free
    base_url: ` + baseURL + `/v1
    api_key_env: EMBEDDING_LLM_API_KEY
    capabilities: [embeddings]
    embedding_dimensions: 1024
`
	path := filepath.Join(t.TempDir(), "models.yaml")
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))
	return path
}

func TestEmbed_EndToEnd_OmittedDimensionReachesTheWireWithNoOverride(t *testing.T) {
	var wireBody map[string]any
	var wireAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		wireAuth = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&wireBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		values := make([]float32, 1024)
		for i := range values {
			values[i] = float32(i%7) / 10
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "liquid/lfm-2.5-embedding-350m:free",
			"data":  []map[string]any{{"embedding": values, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 11},
		})
	}))
	defer srv.Close()

	t.Setenv("EMBEDDING_LLM_API_KEY", "openrouter-e2e-key")
	reg, err := modelregistry.Load(writeE2ERegistry(t, srv.URL))
	require.NoError(t, err)
	resolver, err := registryadapter.NewResolver(reg)
	require.NoError(t, err)

	// The real adapter, constructed with an endpoint it must NOT use and a
	// family key it must NOT send.
	client, err := openai.New(openai.Config{
		HTTPClient: srv.Client(),
		Secrets:    e2eSecretClient{familyKey: "wrong-family-key"},
		Endpoint:   "https://api.openai.com/v1",
	})
	require.NoError(t, err)

	outbox := &fakeEmbedOutbox{}
	svc, err := domain.NewService(domain.ServiceConfig{
		Vendors:        []domain.VendorClient{&fakeVendor{family: domain.VendorFamilyOpenAI}},
		Armor:          &fakeArmor{},
		Budget:         &fakeBudget{},
		Outbox:         outbox,
		Policies:       &fakePolicy{},
		GatewayVersion: "chora-model-gateway:e2e",
		Embedders:      []domain.EmbeddingClient{client},
		Models:         resolver,
		Now:            func() time.Time { return time.Unix(1754500000, 0) },
		NewID:          func() string { return "01988888-8888-7888-8888-888888888888" },
	})
	require.NoError(t, err)

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"
	// The caller OMITS the dimension: the gateway must not invent one.
	req.OutputDimensions = 0

	resp, err := svc.Embed(context.Background(), req)
	require.NoError(t, err)

	// No dimensions parameter on the wire, because the resolved model takes
	// none — the whole point of the correction.
	_, sent := wireBody["dimensions"]
	assert.False(t, sent, "an omitted dimension must not become a dimensions parameter; wire body was %v", wireBody)
	assert.Equal(t, "liquid/lfm-2.5-embedding-350m:free", wireBody["model"])
	assert.Equal(t, "Bearer openrouter-e2e-key", wireAuth,
		"the resolved entry's credential must reach the wire, not the vendor family's key")

	// The model's declared 1024-value vector is accepted and ledgered.
	require.Len(t, resp.Values, 1024)
	assert.Equal(t, int64(11), resp.Usage.InputTokens)
	require.Len(t, outbox.events, 1)
	assert.Equal(t, string(domain.VendorFamilyOpenAI), outbox.events[0].Vendor)
}

// The same end-to-end path refuses a response of the wrong length: a
// 768-element vector for a model configured as 1024 never reaches the caller.
func TestEmbed_EndToEnd_WrongLengthVectorIsRefused(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		values := make([]float32, 768)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "liquid/lfm-2.5-embedding-350m:free",
			"data":  []map[string]any{{"embedding": values, "index": 0}},
			"usage": map[string]any{"prompt_tokens": 11},
		})
	}))
	defer srv.Close()

	t.Setenv("EMBEDDING_LLM_API_KEY", "openrouter-e2e-key")
	reg, err := modelregistry.Load(writeE2ERegistry(t, srv.URL))
	require.NoError(t, err)
	resolver, err := registryadapter.NewResolver(reg)
	require.NoError(t, err)

	client, err := openai.New(openai.Config{
		HTTPClient: srv.Client(),
		Secrets:    e2eSecretClient{familyKey: "family-key"},
		Endpoint:   srv.URL,
	})
	require.NoError(t, err)

	outbox := &fakeEmbedOutbox{}
	svc, err := domain.NewService(domain.ServiceConfig{
		Vendors:        []domain.VendorClient{&fakeVendor{family: domain.VendorFamilyOpenAI}},
		Armor:          &fakeArmor{},
		Budget:         &fakeBudget{},
		Outbox:         outbox,
		Policies:       &fakePolicy{},
		GatewayVersion: "chora-model-gateway:e2e",
		Embedders:      []domain.EmbeddingClient{client},
		Models:         resolver,
		Now:            func() time.Time { return time.Unix(1754500000, 0) },
		NewID:          func() string { return "01988888-8888-7888-8888-888888888888" },
	})
	require.NoError(t, err)

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"

	_, err = svc.Embed(context.Background(), req)
	require.Error(t, err)
	var dimErr *domain.DimensionMismatchError
	require.ErrorAs(t, err, &dimErr)
	assert.Equal(t, int32(1024), dimErr.Expected)
	assert.Equal(t, int32(768), dimErr.Actual)
	assert.Len(t, outbox.events, 0, "a wrong-length vector must never be ledgered")
}
