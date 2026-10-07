# CI/CD push-trigger evidence log

Each dated line below is a benign touch commit whose push to main fires
`chora-model-gateway-main-dev` (path filter `services/chora-model-gateway/**`).
The commit SHA of each line IS the build's trigger cause; the evidence pack
references it in every capture caption (CHO-2371).

- 2026-07-27 CHO-2371 push-trigger evidence sweep fire
- 2026-07-27 CHO-2371 retry fire: transient sca-govulncheck machinery failure at 3eb5242 (clean in-container repro)
- 2026-07-27 CHO-2371 retry 2: root cause was GO-2026-5932 (x/crypto openpgp deprecation advisory, module-level only) landing as NEW vs baseline; baselined with rationale, exact gate script PASSes in-container
