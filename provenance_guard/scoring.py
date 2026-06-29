"""Confidence scoring: combine the two signals into one calibrated verdict.

Single source of truth for the contract described in planning.md
"Uncertainty representation".
"""

from __future__ import annotations

from . import config
from .labels import make_label
from .signals import llm_signal, stylometric_signal


def combine(llm_score: float | None, stylo_score: float) -> dict:
    """Fuse signal AI-likelihoods into ai_likelihood + confidence + attribution.

    ai_likelihood = weighted avg of available signals (1.0 = AI).
    confidence    = max(p, 1-p) in [0.5, 1.0] = probability mass behind the verdict;
                    0.5 means a coin flip ("I don't know").
    attribution   = band on ai_likelihood (asymmetric: high bar to call AI).
    """
    if llm_score is None:
        # Degraded mode: stylometry only. Renormalize weight to 1.0.
        ai_likelihood = stylo_score
        degraded = True
    else:
        ai_likelihood = config.W_LLM * llm_score + config.W_STYLO * stylo_score
        degraded = False

    ai_likelihood = round(max(0.0, min(1.0, ai_likelihood)), 4)
    confidence = round(max(ai_likelihood, 1.0 - ai_likelihood), 4)

    if ai_likelihood >= config.AI_THRESHOLD:
        attribution = "likely_ai"
    elif ai_likelihood <= config.HUMAN_THRESHOLD:
        attribution = "likely_human"
    else:
        attribution = "uncertain"

    # In degraded mode we are less sure than the math implies (we lost a whole
    # signal), so we discount confidence toward 0.5 to communicate that honestly.
    if degraded:
        confidence = round(0.5 + (confidence - 0.5) * 0.7, 4)

    return {
        "ai_likelihood": ai_likelihood,
        "confidence": confidence,
        "attribution": attribution,
        "degraded": degraded,
    }


def classify(text: str, *, client=None) -> dict:
    """Run the full pipeline on text and return a complete, response-ready result."""
    llm = llm_signal(text, client=client)
    stylo = stylometric_signal(text)

    combined = combine(llm["llm_score"], stylo["stylo_score"])
    label = make_label(combined["attribution"], combined["confidence"])

    return {
        "attribution": combined["attribution"],
        "confidence": combined["confidence"],
        "ai_likelihood": combined["ai_likelihood"],
        "label": label,
        "signals": {
            "llm": {
                "score": llm["llm_score"],
                "available": llm["available"],
                "reasoning": llm["reasoning"],
                "weight": config.W_LLM,
            },
            "stylometric": {
                "score": stylo["stylo_score"],
                "weight": config.W_STYLO,
                "breakdown": stylo["breakdown"],
                "note": stylo["note"],
            },
        },
        "degraded_mode": combined["degraded"],
    }
