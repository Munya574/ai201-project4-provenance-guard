"""Two independent detection signals.

Signal 1 (llm_signal): semantic/holistic read via Groq.
Signal 2 (stylometric_signal): structural statistics in pure Python.

Each returns an AI-likelihood in [0, 1] (1.0 = looks fully AI-generated) plus a
breakdown so decisions are explainable and auditable.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from typing import Optional

from . import config

# ---------------------------------------------------------------------------
# Shared text utilities
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s+|$)")
_WORD_RE = re.compile(r"[A-Za-z']+")

# Default AI "tell" phrases. Lowercased substring match. Not exhaustive — a
# documented, deterministic heuristic, not a guarantee.
_AI_PHRASES = (
    "it is important to note",
    "it's important to note",
    "it is worth noting",
    "in conclusion",
    "furthermore",
    "moreover",
    "additionally",
    "in today's world",
    "in the realm of",
    "navigate the complexities",
    "navigating the complexities",
    "delve into",
    "delving into",
    "tapestry",
    "a testament to",
    "plays a crucial role",
    "plays a vital role",
    "when it comes to",
    "on the other hand",
    "ultimately",
    "as a result",
)

_CONTRACTION_RE = re.compile(
    r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", re.IGNORECASE
)


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return parts


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _scale(value: float, low: float, high: float) -> float:
    """Map value to [0,1] linearly, clamped. value<=low -> 0, value>=high -> 1."""
    if high == low:
        return 0.0
    return _clamp((value - low) / (high - low))


# ---------------------------------------------------------------------------
# Signal 2 — Stylometric heuristics
# ---------------------------------------------------------------------------

def stylometric_signal(text: str) -> dict:
    """Structural AI-likelihood from measurable text statistics.

    Returns: {"stylo_score": float, "breakdown": {...}, "note": str|None}

    Sub-metrics (each mapped to an AI-likelihood contribution in [0,1]):
      - burstiness:  low sentence-length variation -> more AI-like
      - lexical_uniformity: moving-average TTR near the "machine" mid-band -> AI
      - casual_markers: contractions + casual punctuation -> more human (inverted)
      - ai_phrases: density of cliché transition/filler phrases -> more AI
    """
    words = _words(text)
    sentences = _sentences(text)
    n_words = len(words)
    note: Optional[str] = None

    if n_words == 0:
        return {"stylo_score": 0.5, "breakdown": {}, "note": "empty text"}

    # --- burstiness: coefficient of variation of sentence lengths -----------
    sent_lengths = [len(_words(s)) for s in sentences if _words(s)]
    if len(sent_lengths) >= 2 and statistics.mean(sent_lengths) > 0:
        cv = statistics.pstdev(sent_lengths) / statistics.mean(sent_lengths)
    else:
        cv = 0.5  # not enough sentences to judge -> neutral-ish
    # Human prose CV is typically ~0.5+; very uniform AI prose ~0.2-0.35.
    # Low CV -> high AI-likelihood, so invert.
    burstiness_ai = 1.0 - _scale(cv, 0.2, 0.7)

    # --- lexical diversity: moving-average type-token ratio (length-robust) --
    mattr = _moving_avg_ttr(words, window=min(50, n_words))
    # Very low (<0.55) or very high (>0.85) diversity reads human (casual repeats
    # or rich vocabulary). The flat mid-band (~0.65-0.8) reads more machine-like.
    if mattr <= 0.55 or mattr >= 0.88:
        lexical_ai = 0.2
    else:
        # peak AI-likelihood around 0.72
        lexical_ai = 1.0 - abs(mattr - 0.72) / 0.16
        lexical_ai = _clamp(lexical_ai)

    # --- casual markers: contractions + informal punctuation ----------------
    contractions = len(_CONTRACTION_RE.findall(text))
    casual_punct = len(re.findall(r"(\.\.\.|!|\?|—|--)", text))
    allcaps = len(re.findall(r"\b[A-Z]{2,}\b", text))
    casual_rate = (contractions + casual_punct + allcaps) / max(1, n_words / 20)
    # More casual markers -> more human -> lower AI-likelihood.
    casual_ai = 1.0 - _scale(casual_rate, 0.0, 3.0)

    # --- AI cliché phrase density -------------------------------------------
    low = text.lower()
    phrase_hits = sum(low.count(p) for p in _AI_PHRASES)
    phrase_density = phrase_hits / max(1, n_words / 100)  # hits per 100 words
    ai_phrases_ai = _scale(phrase_density, 0.0, 2.0)

    # --- combine sub-metrics ------------------------------------------------
    sub = {
        "burstiness_ai": (burstiness_ai, 0.35),
        "lexical_ai": (lexical_ai, 0.20),
        "casual_ai": (casual_ai, 0.25),
        "ai_phrases_ai": (ai_phrases_ai, 0.20),
    }
    stylo_score = sum(v * w for (v, w) in sub.values())

    # --- short-text damping: pull toward 0.5 when stats are unreliable ------
    if n_words < config.MIN_RELIABLE_WORDS:
        factor = n_words / config.MIN_RELIABLE_WORDS
        stylo_score = 0.5 + (stylo_score - 0.5) * factor
        note = (
            f"short text ({n_words} words < {config.MIN_RELIABLE_WORDS}); "
            "stylometric score damped toward uncertain"
        )

    stylo_score = round(_clamp(stylo_score), 4)

    breakdown = {
        "word_count": n_words,
        "sentence_count": len(sent_lengths),
        "sentence_length_cv": round(cv, 4),
        "mattr": round(mattr, 4),
        "contractions": contractions,
        "casual_punct": casual_punct,
        "ai_phrase_hits": phrase_hits,
        "sub_scores": {k: round(v, 4) for k, (v, _w) in sub.items()},
    }
    return {"stylo_score": stylo_score, "breakdown": breakdown, "note": note}


def _moving_avg_ttr(words: list[str], window: int = 50) -> float:
    """Moving-Average Type-Token Ratio. Length-robust vocabulary diversity."""
    if not words:
        return 0.0
    if len(words) <= window:
        return len(set(words)) / len(words)
    ratios = []
    for i in range(len(words) - window + 1):
        chunk = words[i : i + window]
        ratios.append(len(set(chunk)) / window)
    return statistics.mean(ratios)


# ---------------------------------------------------------------------------
# Signal 1 — LLM classifier (Groq)
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = (
    "You are a careful AI-content forensic analyst for a creative writing "
    "platform. Given a piece of text, estimate the probability that it was "
    "generated by an AI language model rather than written by a human. "
    "Weigh semantic flatness, generic 'helpful assistant' voice, unnaturally "
    "even structure, and lack of lived specificity as AI signals; weigh "
    "idiosyncratic voice, concrete personal detail, and natural irregularity "
    "as human signals. Be conservative: do NOT call human work AI without "
    "strong evidence. Respond with STRICT JSON only, no prose, in the form: "
    '{"ai_likelihood": <number 0..1>, "reasoning": "<one short sentence>"}'
)


def llm_signal(text: str, *, client=None) -> dict:
    """Semantic AI-likelihood via Groq.

    Returns: {"llm_score": float|None, "reasoning": str, "available": bool}
    Degrades gracefully: if the key is missing or the call fails, llm_score is
    None and available is False (the caller reweights to stylometry only).
    """
    if not config.GROQ_API_KEY:
        return {
            "llm_score": None,
            "reasoning": "Groq API key not configured; LLM signal skipped.",
            "available": False,
        }

    try:
        if client is None:
            from groq import Groq  # imported lazily so the app runs without the SDK call path

            client = Groq(api_key=config.GROQ_API_KEY)

        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        score = _clamp(float(data.get("ai_likelihood")))
        reasoning = str(data.get("reasoning", "")).strip() or "(no reasoning returned)"
        return {"llm_score": round(score, 4), "reasoning": reasoning, "available": True}
    except Exception as exc:  # noqa: BLE001 — degrade gracefully on any failure
        return {
            "llm_score": None,
            "reasoning": f"LLM signal failed: {type(exc).__name__}: {exc}",
            "available": False,
        }
