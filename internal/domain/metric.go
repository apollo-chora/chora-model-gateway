package domain

// =============================================================================
// Metric cardinality guard.
//
// The gateway emits OTel traces and structured logs. When metrics are added,
// the labels MUST be bounded — a metric label carrying a raw model ID, tenant
// ID, GCID, prompt text, upstream error body, URL, or registry metadata would
// create a cardinality explosion (one time series per unique value) that
// overwhelms the metrics backend and leaks sensitive data.
//
// The safe pattern is:
//
//	// GOOD — bounded label values
//	counter.Add(ctx, 1, metric.WithAttributes(
//		attribute.String("vendor", string(policy.Vendor)),
//		attribute.String("finish_reason", resp.FinishReason.String()),
//	))
//
//	// BAD — unbounded label values
//	counter.Add(ctx, 1, metric.WithAttributes(
//		attribute.String("model_id", string(req.LogicalModelID)),  // unbounded
//		attribute.String("tenant_id", req.TenantID),               // unbounded
//		attribute.String("prompt", req.Prompt),                    // unbounded + sensitive
//	))
//
// The safe label values are: vendor family, finish reason, armor verdict,
// accounting state, and other enum-like bounded sets. The unsafe values are:
// raw model IDs, tenant IDs, GCIDs, prompts, upstream error bodies, URLs, and
// registry metadata.
//
// This file documents the constraint; the enforcement is by code review.
// =============================================================================

// SafeMetricLabels returns the bounded, safe label values for a metric event.
// Use this when adding metrics to ensure the labels are bounded and do not
// leak sensitive data.
type SafeMetricLabels struct {
	// Vendor is the resolved vendor family (bounded: one of the VendorFamily
	// enum values).
	Vendor string
	// FinishReason is the finish reason (bounded: one of the FinishReason
	// enum values).
	FinishReason string
	// ArmorPre is the PRE armor verdict (bounded: one of the ArmorVerdict
	// enum values).
	ArmorPre string
	// ArmorPost is the POST armor verdict (bounded: one of the ArmorVerdict
	// enum values).
	ArmorPost string
	// AccountingState is the accounting state (bounded: one of the
	// AccountingState enum values).
	AccountingState string
	// FailureClass is the failure class (bounded: one of the FailureClass
	// enum values).
	FailureClass string
}

// IsSafeMetricValue reports whether a string value is safe to use as a metric
// label value. A value is safe if it is one of the bounded enum-like values
// (vendor family, finish reason, armor verdict, accounting state, failure
// class). A value is unsafe if it is a raw model ID, tenant ID, GCID, prompt,
// upstream error body, URL, or registry metadata.
//
// This is a heuristic check for code review and testing — it is not a
// runtime enforcement mechanism.
func IsSafeMetricValue(value string) bool {
	// Bounded enum-like values are safe.
	switch value {
	case
		// Vendor families
		string(VendorFamilyVertexGemini),
		string(VendorFamilyVertexGemma),
		string(VendorFamilyOpenAI),
		string(VendorFamilyAnthropic),
		// Finish reasons
		FinishReasonUnspecified.String(),
		FinishReasonComplete.String(),
		FinishReasonMaxTokens.String(),
		FinishReasonModelArmorBlock.String(),
		FinishReasonBudgetBlock.String(),
		FinishReasonVendorError.String(),
		FinishReasonManaBlock.String(),
		// Armor verdicts
		ArmorVerdictUnspecified.String(),
		ArmorVerdictAllow.String(),
		ArmorVerdictBlock.String(),
		ArmorVerdictSanitise.String(),
		ArmorVerdictError.String(),
		ArmorVerdictBypassed.String(),
		// Accounting states
		AccountingUnspecified.String(),
		AccountingClaimed.String(),
		AccountingInFlight.String(),
		AccountingSucceeded.String(),
		AccountingAccounted.String(),
		AccountingFailed.String(),
		// Failure classes
		FailureUnknown.String(),
		FailureTimeout.String(),
		FailureRateLimited.String(),
		FailureProvider5xx.String(),
		FailureInvalidCredentials.String(),
		FailureMalformedRequest.String(),
		FailureUnsupportedCapability.String(),
		FailurePrePolicyRejection.String(),
		FailurePostPolicyRejection.String(),
		FailureBudgetExhausted.String(),
		FailureLedgerFailure.String(),
		FailureInternalInvariant.String():
		return true
	}
	return false
}
