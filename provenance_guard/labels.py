"""Reader-facing transparency labels.

Three variants keyed off `attribution`. Text matches planning.md verbatim
(with the confidence percentage interpolated). Non-accusatory by design: an AI
verdict says "signals suggest", never "you cheated".
"""

from __future__ import annotations

_VARIANTS = {
    "likely_ai": {
        "headline": "🤖 Likely AI-generated",
        "body": (
            "Our analysis suggests this text was probably created with the help "
            "of AI (confidence: {pct}%). This is an automated signal, not a "
            "certainty. If you're the creator and disagree, you can appeal this label."
        ),
    },
    "likely_human": {
        "headline": "✍️ Likely human-written",
        "body": (
            "Our analysis found no strong signs of AI generation in this text "
            "(confidence: {pct}%). No label is perfect, but this reads as "
            "human-written work."
        ),
    },
    "uncertain": {
        "headline": "❓ Authorship unclear",
        "body": (
            "Our analysis couldn't confidently determine whether this text was "
            "written by a human or AI (confidence: {pct}%). We're showing this "
            "honestly rather than guessing. Treat the origin as unverified."
        ),
    },
}


def make_label(attribution: str, confidence: float) -> dict:
    """Map an attribution band + confidence to a displayable label.

    Returns {"variant", "headline", "text"} where text is the full label string.
    """
    variant = _VARIANTS.get(attribution)
    if variant is None:
        raise ValueError(f"unknown attribution: {attribution!r}")
    pct = round(confidence * 100)
    return {
        "variant": attribution,
        "headline": variant["headline"],
        "text": f"{variant['headline']} — {variant['body'].format(pct=pct)}",
    }
