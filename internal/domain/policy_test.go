package domain

import "testing"

// ---------------------------------------------------------------------------
// Endpoint joining
//
// resolveEndpoint is the only place a registry entry's base_url and its path
// are combined. Every error here is a request that goes to the wrong host or
// the wrong path, so the rules are pinned individually rather than through one
// happy-path case.
// ---------------------------------------------------------------------------

func TestTargetModel_ChatCompletionsURL(t *testing.T) {
	cases := []struct {
		name string
		base string
		path string
		want string
	}{
		{
			name: "plain join",
			base: "https://api.openai.com/v1",
			path: "/chat/completions",
			want: "https://api.openai.com/v1/chat/completions",
		},
		{
			// A trailing slash on base_url is the most common registry typo and
			// must not produce "https://host/v1//chat/completions", which some
			// gateways route differently from the single-slash form.
			name: "trailing slash on the base is trimmed",
			base: "https://api.openai.com/v1/",
			path: "/chat/completions",
			want: "https://api.openai.com/v1/chat/completions",
		},
		{
			name: "many trailing slashes are all trimmed",
			base: "https://api.openai.com/v1///",
			path: "/chat/completions",
			want: "https://api.openai.com/v1/chat/completions",
		},
		{
			// A relative path is joined with exactly one separator, so the
			// registry author does not have to remember to add a slash.
			name: "relative path gets a separator",
			base: "https://api.openai.com/v1",
			path: "generate",
			want: "https://api.openai.com/v1/generate",
		},
		{
			name: "empty path falls back to the default",
			base: "https://api.openai.com/v1",
			path: "",
			want: "https://api.openai.com/v1/chat/completions",
		},
		{
			// An absolute path is the "fully custom endpoint" escape hatch: it
			// is used verbatim and base_url is ignored entirely, because the
			// whole point is that the target lives somewhere else.
			name: "absolute path overrides the base",
			base: "https://ignored.example.com/v1",
			path: "https://llm.internal:8443/generate",
			want: "https://llm.internal:8443/generate",
		},
		{
			// An empty base means a same-origin relative call — how a gateway
			// fronted by a single reverse proxy expresses "the LLM is next
			// door". The result is the bare path, which is not a usable
			// absolute URL, and that is deliberate: the caller is expected to
			// resolve it against the gateway's own origin.
			name: "empty base yields the bare path",
			base: "",
			path: "/chat/completions",
			want: "/chat/completions",
		},
		{
			name: "empty base and empty path yields the default path",
			base: "",
			path: "",
			want: "/chat/completions",
		},
		{
			name: "empty base with a relative path",
			base: "",
			path: "generate",
			want: "generate",
		},
		{
			name: "base with no path is still a valid root",
			base: "https://llm.internal",
			path: "/chat/completions",
			want: "https://llm.internal/chat/completions",
		},
		{
			name: "a local server on a port",
			base: "http://host.docker.internal:11434/v1",
			path: "/chat/completions",
			want: "http://host.docker.internal:11434/v1/chat/completions",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := TargetModel{BaseURL: tc.base, ChatCompletionsPath: tc.path}.ChatCompletionsURL()
			if got != tc.want {
				t.Errorf("ChatCompletionsURL = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestTargetModel_MessagesURL(t *testing.T) {
	cases := []struct {
		name string
		base string
		path string
		want string
	}{
		{
			// The Anthropic default is /v1/messages, NOT OpenAI's
			// /chat/completions. Sending an anthropic entry to the OpenAI path
			// is a 404 from the provider on every call.
			name: "default messages path",
			base: "https://api.anthropic.com",
			path: "",
			want: "https://api.anthropic.com/v1/messages",
		},
		{
			name: "trailing slash on the base is trimmed",
			base: "https://api.anthropic.com/",
			path: "/v1/messages",
			want: "https://api.anthropic.com/v1/messages",
		},
		{
			name: "configured path wins",
			base: "https://proxy.test",
			path: "/anthropic/v1/messages",
			want: "https://proxy.test/anthropic/v1/messages",
		},
		{
			name: "absolute path overrides the base",
			base: "https://ignored.test",
			path: "https://llm.internal:8443/v1/messages",
			want: "https://llm.internal:8443/v1/messages",
		},
		{
			name: "empty base yields the bare default path",
			base: "",
			path: "",
			want: "/v1/messages",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := TargetModel{BaseURL: tc.base, MessagesPath: tc.path}.MessagesURL()
			if got != tc.want {
				t.Errorf("MessagesURL = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestTargetModel_ImagesAndEmbeddingsURL(t *testing.T) {
	cases := []struct {
		name           string
		base           string
		imagesPath     string
		embeddingsPath string
		wantImages     string
		wantEmbeddings string
	}{
		{
			name:           "defaults",
			base:           "https://api.openai.com/v1",
			wantImages:     "https://api.openai.com/v1/images/generations",
			wantEmbeddings: "https://api.openai.com/v1/embeddings",
		},
		{
			name:           "trailing slash on the base is trimmed",
			base:           "https://api.openai.com/v1/",
			wantImages:     "https://api.openai.com/v1/images/generations",
			wantEmbeddings: "https://api.openai.com/v1/embeddings",
		},
		{
			// The override applies per-surface: a gateway that proxies images at
			// a different host must not have its embeddings path redirected
			// too.
			name:           "per-surface overrides are independent",
			base:           "https://api.openai.com/v1",
			imagesPath:     "https://images.internal/gen",
			embeddingsPath: "/embed",
			wantImages:     "https://images.internal/gen",
			wantEmbeddings: "https://api.openai.com/v1/embed",
		},
		{
			name:           "empty base yields bare paths",
			base:           "",
			wantImages:     "/images/generations",
			wantEmbeddings: "/embeddings",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tgt := TargetModel{
				BaseURL:        tc.base,
				ImagesPath:     tc.imagesPath,
				EmbeddingsPath: tc.embeddingsPath,
			}
			if got := tgt.ImagesURL(); got != tc.wantImages {
				t.Errorf("ImagesURL = %q, want %q", got, tc.wantImages)
			}
			if got := tgt.EmbeddingsURL(); got != tc.wantEmbeddings {
				t.Errorf("EmbeddingsURL = %q, want %q", got, tc.wantEmbeddings)
			}
		})
	}
}

// TestTargetModel_PrimaryDispatchURL pins the provider-aware switch. The boot
// log prints this, so an operator sees the URL that will really be called; an
// OpenAI-shaped guess at an Anthropic entry sends them chasing the wrong path.
func TestTargetModel_PrimaryDispatchURL(t *testing.T) {
	anthropic := TargetModel{
		Vendor:       VendorFamilyAnthropic,
		BaseURL:      "https://api.anthropic.com",
		MessagesPath: "",
		// Deliberately set: the Anthropic adapter never reads this field, so
		// its presence must not change the answer.
		ChatCompletionsPath: "/chat/completions",
	}
	if got, want := anthropic.PrimaryDispatchURL(), "https://api.anthropic.com/v1/messages"; got != want {
		t.Errorf("anthropic PrimaryDispatchURL = %q, want %q", got, want)
	}

	openai := TargetModel{
		Vendor:              VendorFamilyOpenAI,
		BaseURL:             "https://api.openai.com/v1",
		MessagesPath:        "/v1/messages", // set but not read by the openai adapter
		ChatCompletionsPath: "",
	}
	if got, want := openai.PrimaryDispatchURL(), "https://api.openai.com/v1/chat/completions"; got != want {
		t.Errorf("openai PrimaryDispatchURL = %q, want %q", got, want)
	}

	// An unregistered family gets the OpenAI shape rather than an empty string:
	// the boot log should show something an operator can compare against the
	// registry, and the service refuses the dispatch separately.
	unknown := TargetModel{Vendor: VendorFamily("no-such-family"), BaseURL: "https://x.test/v1"}
	if got := unknown.PrimaryDispatchURL(); got != "https://x.test/v1/chat/completions" {
		t.Errorf("unknown-family PrimaryDispatchURL = %q", got)
	}
}

// ---------------------------------------------------------------------------
// Absolute-URL detection
// ---------------------------------------------------------------------------

func TestIsAbsoluteURL(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want bool
	}{
		{"https url", "https://api.anthropic.com/v1/messages", true},
		{"http url", "http://llm.internal:8080/generate", true},
		{"https with no path", "https://x", true},
		{"http with no path", "http://x", true},
		{"bare https scheme", "https://", false}, // no host: nothing to join to
		{"bare http scheme", "http://", false},
		{"leading slash", "/v1/messages", false},
		{"relative", "generate", false},
		{"empty", "", false},
		{"a host with no scheme", "api.anthropic.com/v1", false},
		// Anything that merely CONTAINS a scheme is not absolute. This is the
		// distinction resolveEndpoint needs, and it is narrower than
		// config.validateEndpoint, which accepts any parseable URL.
		{"scheme-relative", "//llm.internal/generate", false},
		{"non-http scheme", "grpc://llm.internal:9000", false},
		{"mailto-ish", "mailto:ops@x.test", false},
		{"a path that merely mentions http", "/proxy/http://upstream", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isAbsoluteURL(tc.in); got != tc.want {
				t.Errorf("isAbsoluteURL(%q) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

// TestResolveEndpoint_NonHTTPSchemeIsJoined documents the seam between the two
// absolute-URL tests. config.LoadRegistry accepts any parseable absolute URL
// for a path override, but resolveEndpoint only recognises http:// and
// https://, so a non-http override is concatenated onto base_url and produces a
// nonsense URL rather than being used verbatim or rejected. See the report
// accompanying these tests.
func TestResolveEndpoint_NonHTTPSchemeIsJoined(t *testing.T) {
	got := resolveEndpoint("https://api.openai.com/v1", "grpc://llm.internal:9000/gen", "/chat/completions")
	if got != "https://api.openai.com/v1/grpc://llm.internal:9000/gen" {
		t.Errorf("resolveEndpoint = %q", got)
	}
}

// ---------------------------------------------------------------------------
// Trailing-slash trimming
// ---------------------------------------------------------------------------

func TestTrimTrailingSlash(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"none", "https://x.test/v1", "https://x.test/v1"},
		{"one", "https://x.test/v1/", "https://x.test/v1"},
		{"several", "https://x.test/v1////", "https://x.test/v1"},
		{"only slashes", "///", ""},
		{"single slash", "/", ""},
		{"empty", "", ""},
		{"root", "/", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := trimTrailingSlash(tc.in); got != tc.want {
				t.Errorf("trimTrailingSlash(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

// TestTargetModel_Supports pins the gate. A caller addressing a surface the
// model does not advertise is refused rather than silently downgraded to text,
// so the empty-capabilities default matters: an entry that names no
// capabilities means chat ONLY, not "anything goes".
func TestTargetModel_Supports(t *testing.T) {
	cases := []struct {
		name string
		caps []string
		cap  string
		want bool
		why  string
	}{
		{"empty caps allow chat", nil, CapabilityChat, true, "the default is chat-only"},
		{"empty caps refuse tools", nil, CapabilityTools, false, "an entry that names nothing cannot be assumed to do tools"},
		{"empty caps refuse image", nil, CapabilityImage, false, "silently downgrading an image request to text would be a wrong answer, not a degraded one"},
		{"empty caps refuse embeddings", nil, CapabilityEmbeddings, false, "same, for the embed surface"},
		{"empty caps refuse vision", nil, CapabilityVision, false, "same, for the vision surface"},

		{"declared chat", []string{CapabilityChat}, CapabilityChat, true, ""},
		{"declared tools", []string{CapabilityChat, CapabilityTools}, CapabilityTools, true, ""},
		{"declared vision", []string{CapabilityChat, CapabilityVision}, CapabilityVision, true, ""},
		{"declared image", []string{CapabilityChat, CapabilityImage}, CapabilityImage, true, ""},
		{"declared embeddings", []string{CapabilityEmbeddings}, CapabilityEmbeddings, true, ""},

		{"undeclared among several", []string{CapabilityChat, CapabilityTools}, CapabilityImage, false, "an unrelated capability must not imply the image one"},
		{"tools does not imply vision", []string{CapabilityTools}, CapabilityVision, false, "capabilities are independent claims, not a hierarchy"},
		{"image does not imply embeddings", []string{CapabilityImage}, CapabilityEmbeddings, false, "same, across the vendor surfaces"},

		// The comparison is verbatim, and the registry validator rejects any
		// token outside KnownCapabilities, so a differently-cased spelling
		// cannot arrive through a validated registry. It can arrive through a
		// hand-built TargetModel, where the safe answer is to refuse.
		{"uppercase is not a match", []string{"CHAT"}, CapabilityChat, false, "the comparison is case-sensitive"},
		{"an undeclared unknown capability", []string{CapabilityChat}, "telepathy", false, "a capability that is not on the list is not granted"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := TargetModel{Capabilities: tc.caps}.Supports(tc.cap)
			if got != tc.want {
				t.Errorf("Supports(%q) with caps %v = %v, want %v (%s)", tc.cap, tc.caps, got, tc.want, tc.why)
			}
		})
	}
}

// TestTargetModel_SupportsIsMembershipOnly pins the layer boundary. Supports
// carries no vocabulary of its own: it is a plain membership test, and the
// thing that keeps an unrecognised capability name out of a TargetModel is the
// registry validator (config.LoadRegistry rejects it against
// domain.KnownCapabilities). Documented here so the gap is not mistaken for a
// second line of defence.
func TestTargetModel_SupportsIsMembershipOnly(t *testing.T) {
	tgt := TargetModel{Capabilities: []string{"telepathy"}}
	if !tgt.Supports("telepathy") {
		t.Error("Supports is membership-only; an unvalidated token answers for itself")
	}
	if tgt.Supports(CapabilityChat) {
		t.Error("an unvalidated token list must not imply chat")
	}
}

// TestTargetModel_SupportsDuplicateCapabilities: a registry file that names the
// same capability twice is harmless — the gate answers a yes/no question and
// the duplicate adds nothing. It is pinned so a future "reject duplicates"
// change is a deliberate one.
func TestTargetModel_SupportsDuplicateCapabilities(t *testing.T) {
	tgt := TargetModel{Capabilities: []string{CapabilityChat, CapabilityChat}}
	if !tgt.Supports(CapabilityChat) {
		t.Error("a duplicated capability changed the answer to no")
	}
}
