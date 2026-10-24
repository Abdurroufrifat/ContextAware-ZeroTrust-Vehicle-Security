"""Research decision explanations; do not conceal the gate that rejected a request."""
def fusion_reason(policy,candidate_ok,candidate_status,radar_ok,radar_status,agreement,context_accept):
    if policy.startswith("existing_"): return "existing_context_policy"
    if not candidate_ok: return "candidate_"+candidate_status
    if not radar_ok: return "radar_"+radar_status
    if not agreement: return "radar_disagreement"
    if policy=="context_no_ood_plus_radar" and not context_accept: return "learned_context_rejection"
    return "radar_agreement"
