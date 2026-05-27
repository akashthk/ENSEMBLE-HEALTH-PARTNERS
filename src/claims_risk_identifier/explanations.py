from claims_risk_identifier.models import assign_risk_tier


def infer_claim_risk_factors(row):
    """Generate simple, grounded claim-level risk factors from field values."""
    factors = []

    if row.get("missing_documentation_flag", 0) == 1:
        factors.append("missing documentation")
    if row.get("eligibility_verified", 1) == 0:
        factors.append("eligibility not verified")
    if row.get("auth_missing_when_required", 0) == 1:
        factors.append("prior authorization required but not on file")
    if row.get("referral_missing_when_required", 0) == 1:
        factors.append("referral required but not present")
    if row.get("is_in_network", 1) == 0:
        factors.append("provider is out of network")
    if row.get("late_submission_30d", 0) == 1:
        factors.append("claim submitted 30 or more days after service")
    if row.get("payment_to_billed_ratio", 1) < 0.4:
        factors.append("low expected payment compared with billed amount")
    if row.get("complex_claim", 0) == 1:
        factors.append("complex claim with many procedures or diagnoses")
    if row.get("admin_issue_count", 0) >= 3:
        factors.append("multiple administrative risk issues")
    if not factors:
        factors.append("overall claim pattern similar to previously denied claims")

    return factors[:3]


def build_llm_prompt_for_claim(claim_row, denial_probability, top_risk_factors):
    """Build the LLM prompt for a single claim explanation."""
    return f"""
You are helping a hospital claims analyst review denial risk before claim submission.

Write a short explanation for the claim below.

Rules:
- Use only the claim fields and risk factors provided.
- Do not invent facts.
- Use plain English.
- Mention that this is a risk estimate, not a guaranteed denial.
- Include one specific recommended action.
- Keep the explanation to 2-3 sentences.

Claim ID: {claim_row["claim_id"]}
Predicted denial probability: {denial_probability:.3f}

Key risk factors:
{", ".join(top_risk_factors)}

Claim fields:
payer_id: {claim_row.get("payer_id")}
payer_type: {claim_row.get("payer_type")}
visit_type: {claim_row.get("visit_type")}
total_billed: {claim_row.get("total_billed")}
expected_payment: {claim_row.get("expected_payment")}
num_procedures: {claim_row.get("num_procedures")}
num_diagnoses: {claim_row.get("num_diagnoses")}
prior_auth_required: {claim_row.get("prior_auth_required")}
has_prior_auth: {claim_row.get("has_prior_auth")}
is_in_network: {claim_row.get("is_in_network")}
days_to_submit: {claim_row.get("days_to_submit")}
missing_documentation_flag: {claim_row.get("missing_documentation_flag")}
eligibility_verified: {claim_row.get("eligibility_verified")}
referral_required: {claim_row.get("referral_required")}
referral_present: {claim_row.get("referral_present")}
service_month: {claim_row.get("service_month")}
""".strip()


def create_manual_explanation(claim_row, denial_probability, top_risk_factors):
    """Deterministic fallback explanation generator."""
    factor_text = ", ".join(top_risk_factors)
    if len(top_risk_factors) == 1:
        factor_text = top_risk_factors[0]

    return (
        f"This claim has an estimated denial risk of {denial_probability:.1%}, "
        f"mainly because of {factor_text}. "
        "This is a risk estimate, not a guarantee of denial; before submission, "
        "review these items and correct or document them where possible."
    )


def generate_openai_explanation(
    claim_row,
    denial_probability,
    top_risk_factors,
    api_key,
    model="gpt-5.5-medium",
):
    """Generate a grounded claim-risk explanation using OpenAI when available."""
    try:
        from openai import OpenAI

        prompt = build_llm_prompt_for_claim(
            claim_row=claim_row,
            denial_probability=denial_probability,
            top_risk_factors=top_risk_factors,
        )
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You write concise, factual explanations for healthcare "
                        "claims analysts. Use only the provided claim data. "
                        "Do not invent facts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=160,
        )
        return response.output_text.strip()
    except Exception as error:
        print(f"OpenAI explanation failed for claim {claim_row.get('claim_id')}: {error}")
        return create_manual_explanation(
            claim_row=claim_row,
            denial_probability=denial_probability,
            top_risk_factors=top_risk_factors,
        )


def score_current_claims(
    model,
    current_df,
    feature_cols,
    threshold,
    high_threshold,
    medium_threshold,
    use_openai_explanations=False,
    openai_api_key=None,
    openai_model="gpt-5.5-medium",
    llm_explanation_top_n=10,
):
    """Score current claims and generate explanations."""
    scored_df = current_df.copy()
    scored_df["denial_probability"] = model.predict_proba(scored_df[feature_cols])[:, 1]
    scored_df["predicted_denial"] = (
        scored_df["denial_probability"] >= threshold
    ).astype(int)
    scored_df["risk_tier"] = scored_df["denial_probability"].apply(
        lambda p: assign_risk_tier(p, high_threshold, medium_threshold)
    )
    scored_df = scored_df.sort_values("denial_probability", ascending=False)

    top_llm_claim_ids = set(scored_df.head(llm_explanation_top_n)["claim_id"])
    top_risk_factors_list = []
    explanation_list = []
    prompt_list = []

    for _, row in scored_df.iterrows():
        factors = infer_claim_risk_factors(row)
        prompt = build_llm_prompt_for_claim(row, row["denial_probability"], factors)
        should_use_openai = (
            use_openai_explanations
            and openai_api_key is not None
            and row["claim_id"] in top_llm_claim_ids
        )
        if should_use_openai:
            explanation = generate_openai_explanation(
                row,
                row["denial_probability"],
                factors,
                api_key=openai_api_key,
                model=openai_model,
            )
        else:
            explanation = create_manual_explanation(
                row, row["denial_probability"], factors
            )

        top_risk_factors_list.append("; ".join(factors))
        explanation_list.append(explanation)
        prompt_list.append(prompt)

    scored_df["top_risk_factors"] = top_risk_factors_list
    scored_df["explanation"] = explanation_list
    scored_df["llm_prompt"] = prompt_list

    required_cols = [
        "claim_id",
        "denial_probability",
        "predicted_denial",
        "risk_tier",
        "top_risk_factors",
        "explanation",
    ]
    return scored_df[required_cols], scored_df
