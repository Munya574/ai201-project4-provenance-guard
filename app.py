"""Provenance Guard — Flask API.

Endpoints:
  POST /submit   classify text, persist, log, return verdict + transparency label
  POST /appeal   contest a classification; status -> under_review, logged
  GET  /log      recent structured audit-log entries (grading/documentation)
  GET  /queue    appeal review queue (content under_review)
  GET  /health   liveness + whether the LLM signal is configured
"""

from __future__ import annotations

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from provenance_guard import config, storage
from provenance_guard.scoring import classify

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

storage.init_db()


def _error(message: str, status: int):
    return jsonify({"error": message}), status


@app.route("/submit", methods=["POST"])
@limiter.limit(config.SUBMIT_RATE_LIMIT)
def submit():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    creator_id = (data.get("creator_id") or "").strip()

    if not text:
        return _error("'text' is required and must be non-empty.", 400)
    if not creator_id:
        return _error("'creator_id' is required.", 400)

    result = classify(text)
    content_id = storage.record_classification(creator_id, text, result)

    return jsonify({
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": result["attribution"],
        "confidence": result["confidence"],
        "ai_likelihood": result["ai_likelihood"],
        "label": result["label"],
        "signals": result["signals"],
        "degraded_mode": result["degraded_mode"],
        "status": "classified",
    }), 201


@app.route("/appeal", methods=["POST"])
def appeal():
    data = request.get_json(silent=True) or {}
    content_id = (data.get("content_id") or "").strip()
    reasoning = (data.get("creator_reasoning") or "").strip()

    if not content_id:
        return _error("'content_id' is required.", 400)
    if not reasoning:
        return _error("'creator_reasoning' is required.", 400)

    updated = storage.record_appeal(content_id, reasoning)
    if updated is None:
        return _error(f"No content found for content_id {content_id!r}.", 404)

    return jsonify({
        "content_id": content_id,
        "status": "under_review",
        "message": (
            "Appeal received. This content has been flagged for human review; "
            "its classification is now marked 'under review'."
        ),
        "original_decision": {
            "attribution": updated["attribution"],
            "confidence": updated["confidence"],
            "ai_likelihood": updated["ai_likelihood"],
        },
    }), 200


@app.route("/log", methods=["GET"])
def log():
    limit = request.args.get("limit", default=50, type=int)
    return jsonify({"entries": storage.get_log(limit)}), 200


@app.route("/queue", methods=["GET"])
def queue():
    return jsonify({"under_review": storage.get_review_queue()}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "llm_signal_configured": bool(config.GROQ_API_KEY),
        "model": config.GROQ_MODEL,
    }), 200


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded.",
        "limit": str(e.description),
        "message": "Too many submissions. Please slow down and try again later.",
    }), 429


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
