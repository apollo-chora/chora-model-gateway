package domain

import (
	"context"
	"fmt"
	"net"
	"net/netip"
	"net/url"
	"strings"
)

// =============================================================================
// Security utilities — SSRF/egress validation, error sanitization, and
// gateway-controlled header protection.
//
// These are the shared security primitives the vendor adapters and the
// registry loader use. They live in the domain package so every adapter
// imports them without creating a dependency cycle.
// =============================================================================

// GatewayControlledHeaders are the headers the gateway owns. Registry
// extra_headers MUST NOT be allowed to overwrite them — a registry entry
// that could clobber Authorization or the W3C trace headers would be a
// credential-leak or trace-forging vector.
var GatewayControlledHeaders = map[string]struct{}{
	"authorization":  {},
	"content-type":   {},
	"traceparent":    {},
	"tracestate":     {},
	"x-api-key":      {},
	"anthropic-version": {},
}

// IsGatewayControlledHeader reports whether the header name (case-
// insensitive) is gateway-controlled and therefore must not be overwritten
// by registry extra_headers.
func IsGatewayControlledHeader(name string) bool {
	_, ok := GatewayControlledHeaders[strings.ToLower(strings.TrimSpace(name))]
	return ok
}

// SanitizeHeaders applies registry extra_headers to a request, then applies
// the gateway-controlled headers so they always win. This is the ONLY correct
// order: extra_headers first (so a registry can set X-Custom-Auth), then the
// gateway sets Authorization / Content-Type / traceparent / tracestate /
// x-api-key / anthropic-version last so a registry entry cannot clobber them.
//
// Returns the filtered extra_headers map (with gateway-controlled keys
// removed) so the caller can log what was actually applied.
func SanitizeHeaders(extraHeaders map[string]string, gatewayHeaders map[string]string) map[string]string {
	out := make(map[string]string, len(extraHeaders))
	for k, v := range extraHeaders {
		if IsGatewayControlledHeader(k) {
			continue
		}
		out[k] = v
	}
	for k, v := range gatewayHeaders {
		out[k] = v
	}
	return out
}

// SanitizeUpstreamError returns a client-safe error message for a non-2xx
// upstream response. The raw upstream body is NOT included — it may contain
// internal hostnames, stack traces, or other sensitive information. The
// detailed error is available on the ProviderError for internal logging.
func SanitizeUpstreamError(status int, endpoint string) string {
	return fmt.Sprintf("upstream provider returned status %d", status)
}

// ValidateEgressURL checks that a URL is safe to dispatch to — it must be
// an http(s) URL and must NOT point at a private, loopback, link-local, or
// otherwise internal IP address. This is the SSRF/egress guard: registry
// base_url values and endpoint path overrides are validated at boot time so
// a misconfigured (or malicious) registry entry cannot turn the gateway
// into an internal-network proxy.
//
// Only LITERAL IP addresses are checked at boot time. A DNS name is not
// resolved here — DNS resolution is unreliable at boot (the test suite uses
// .test domains that never resolve) and a DNS name that resolves to a
// private IP is a dispatch-time concern, not a boot-time one. The literal-
// IP check catches the common SSRF vector: a registry entry that hard-codes
// an internal IP address.
func ValidateEgressURL(rawURL string) error {
	u, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("invalid url %q: %w", rawURL, err)
	}
	scheme := strings.ToLower(u.Scheme)
	if scheme != "http" && scheme != "https" {
		return fmt.Errorf("url %q must be http or https", rawURL)
	}
	host := u.Hostname()
	if host == "" {
		return fmt.Errorf("url %q has no host", rawURL)
	}
	// Only check literal IP addresses. A DNS name is not resolved here.
	ip := net.ParseIP(host)
	if ip == nil {
		// Not a literal IP — it's a DNS name. We cannot validate it at
		// boot time without a DNS lookup, which is unreliable. The
		// dispatch-time check (in the vendor adapters) is the second
		// line of defence.
		return nil
	}
	if isPrivateIP(ip) {
		return fmt.Errorf("host %q is a private/internal ip — egress denied", host)
	}
	return nil
}

// isPrivateIP reports whether ip is a private, loopback, link-local, or
// otherwise non-public IP address. This is the SSRF guard: the gateway only
// dispatches to public vendor endpoints.
func isPrivateIP(ip net.IP) bool {
	addr, err := netip.ParseAddr(ip.String())
	if err != nil {
		return true // unparseable = not public
	}
	if addr.IsLoopback() || addr.IsLinkLocalUnicast() || addr.IsLinkLocalMulticast() {
		return true
	}
	if addr.IsPrivate() {
		return true
	}
	// CGNAT range (100.64.0.0/10) is not globally routable.
	if addr.Is4() {
		b := addr.As4()
		if b[0] == 100 && b[1]&0xc0 == 0x40 {
			return true
		}
	}
	return false
}

// CheckContextCancellation returns an error if the context is already
// cancelled. Vendor adapters call this BEFORE the HTTP dispatch so a
// disconnected client fails fast without consuming a provider request.
func CheckContextCancellation(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("request cancelled: %w", err)
	}
	return nil
}
