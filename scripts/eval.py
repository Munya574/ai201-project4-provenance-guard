"""Calibration harness — validates that scoring produces meaningful separation.

Runs the full pipeline on labelled reference inputs and prints per-signal and
combined scores, the attribution band, and the label variant. Use this to
confirm clearly-AI vs clearly-human texts separate and borderline cases land in
'uncertain'. Works with or without a Groq key (degrades to stylometry only).

Run:  python -m scripts.eval
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provenance_guard.scoring import classify  # noqa: E402

CASES = [
    ("Clearly AI-generated",
     "Artificial intelligence represents a transformative paradigm shift in "
     "modern society. It is important to note that while the benefits of AI are "
     "numerous, it is equally essential to consider the ethical implications. "
     "Furthermore, stakeholders across various sectors must collaborate to "
     "ensure responsible deployment."),

    ("Clearly human-written",
     "ok so i finally tried that new ramen place downtown and honestly? "
     "underwhelming. the broth was fine but they put WAY too much sodium in it "
     "and i was thirsty for like three hours after. my friend got the spicy "
     "version and said it was better. probably won't go back unless someone "
     "drags me there"),

    ("Borderline: formal human writing",
     "The relationship between monetary policy and asset price inflation has "
     "been extensively studied in the literature. Central banks face a "
     "fundamental tension between their mandate for price stability and the "
     "unintended consequences of prolonged low interest rates on equity and "
     "real estate valuations."),

    ("Borderline: lightly edited AI output",
     "I've been thinking a lot about remote work lately. There are genuine "
     "tradeoffs — flexibility and no commute on one side, isolation and blurred "
     "work-life boundaries on the other. Studies show productivity varies "
     "widely by individual and role type."),
]


def main() -> None:
    print(f"{'CASE':<34} {'llm':>6} {'stylo':>7} {'ai_lik':>7} {'conf':>6}  band")
    print("-" * 80)
    for name, text in CASES:
        r = classify(text)
        llm = r["signals"]["llm"]["score"]
        stylo = r["signals"]["stylometric"]["score"]
        llm_str = f"{llm:.3f}" if llm is not None else "  n/a"
        print(
            f"{name:<34} {llm_str:>6} {stylo:>7.3f} "
            f"{r['ai_likelihood']:>7.3f} {r['confidence']:>6.3f}  "
            f"{r['attribution']}"
        )
    print()
    if classify("test")["degraded_mode"]:
        print("NOTE: running in DEGRADED mode (no Groq key) — stylometry only.")


if __name__ == "__main__":
    main()
