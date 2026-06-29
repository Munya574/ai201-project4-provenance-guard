"""Central configuration. All tunable knobs live here so the scoring contract
in planning.md has exactly one source of truth in code."""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Groq / LLM signal -------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

# --- Storage -----------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "provenance_guard.db").strip()

# --- Signal combination weights (must sum to 1.0) ----------------------------
# LLM gets more weight: it reasons about meaning. Stylometry is the structural
# check that keeps it honest. See planning.md "Detection signals".
W_LLM = 0.65
W_STYLO = 0.35

# --- Attribution bands on ai_likelihood --------------------------------------
# Asymmetric ON PURPOSE: the AI threshold (0.70) sits higher above the midpoint
# than the human threshold (0.35) sits below it. This widens the "uncertain" band
# on the AI side so borderline text — especially formal/non-native human prose —
# falls into "uncertain" rather than being falsely branded AI. False-positiving a
# human writer is the costliest error on a writing platform (see planning.md).
AI_THRESHOLD = 0.70        # >= this -> likely_ai
HUMAN_THRESHOLD = 0.35     # <= this -> likely_human
# between the two -> uncertain

# --- Rate limiting -----------------------------------------------------------
SUBMIT_RATE_LIMIT = "10 per minute;100 per day"

# --- Short-text handling -----------------------------------------------------
# Below this word count, stylometric statistics are unstable, so we damp the
# score toward 0.5 (uncertain) rather than emit a confident verdict.
MIN_RELIABLE_WORDS = 40
