"""
src/audit/database.py — SQLite-backed audit trail for all predictions and chat interactions.

Provides HIPAA-aware logging of every prediction, chat message, and session.
All data is stored locally in data/audit.db with full query support.

Usage:
    from src.audit.database import AuditDB
    db = AuditDB()
    db.log_prediction(session_id, features, prediction, explanation)
    history = db.get_prediction_history(limit=50)
"""
from __future__ import annotations

import json
import sqlite3
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import AUDIT_DB_PATH, AUDIT_ENABLED

logger = logging.getLogger(__name__)


class AuditDB:
    """SQLite audit database for BreastCare AI."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or AUDIT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a connection with row factory enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id              TEXT PRIMARY KEY,
                    start_time      TEXT NOT NULL,
                    end_time        TEXT,
                    total_predictions INTEGER DEFAULT 0,
                    total_chats     INTEGER DEFAULT 0,
                    model_version   TEXT DEFAULT 'v2.0'
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT NOT NULL,
                    timestamp       TEXT NOT NULL,
                    age             INTEGER NOT NULL,
                    shape           INTEGER NOT NULL,
                    margin          INTEGER NOT NULL,
                    density         INTEGER NOT NULL,
                    probability     REAL NOT NULL,
                    risk_category   TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    dominant_feature TEXT,
                    shap_values     TEXT,
                    model_used      TEXT DEFAULT 'lightgbm',
                    pipeline_mode   TEXT DEFAULT 'single',
                    confidence_band TEXT,  -- legacy column, no longer written (the old band was not a real interval)
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS chat_logs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT NOT NULL,
                    timestamp       TEXT NOT NULL,
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    sources         TEXT,
                    safety_filters_fired INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_predictions_session
                    ON predictions(session_id);
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
                    ON predictions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_chat_session
                    ON chat_logs(session_id);
            """)
        logger.info("Audit database initialized at %s", self.db_path)

    # ── Session management ──────────────────────────────────────

    def create_session(self) -> str:
        """Create a new audit session. Returns session ID."""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, start_time) VALUES (?, ?)",
                (session_id, now),
            )
        return session_id

    def close_session(self, session_id: str) -> None:
        """Mark a session as ended."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET end_time = ? WHERE id = ?",
                (now, session_id),
            )

    # ── Prediction logging ──────────────────────────────────────

    def log_prediction(
        self,
        session_id: str,
        features: dict,
        prediction: dict,
        explanation: dict | None = None,
        model_used: str = "lightgbm",
        pipeline_mode: str = "single",
    ) -> int:
        """
        Log a prediction to the audit trail.

        Returns the prediction row ID.
        """
        if not AUDIT_ENABLED:
            return -1

        now = datetime.now(timezone.utc).isoformat()
        shap_json = json.dumps(explanation.get("feature_attributions", {})) if explanation else "{}"
        dominant = explanation.get("dominant_feature", "") if explanation else ""

        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO predictions
                   (session_id, timestamp, age, shape, margin, density,
                    probability, risk_category, predicted_class,
                    dominant_feature, shap_values, model_used,
                    pipeline_mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id, now,
                    features["age"], features["shape"],
                    features["margin"], features["density"],
                    prediction["probability"],
                    prediction["risk_category"],
                    prediction["predicted_class"],
                    dominant, shap_json, model_used,
                    pipeline_mode,
                ),
            )
            conn.execute(
                "UPDATE sessions SET total_predictions = total_predictions + 1 WHERE id = ?",
                (session_id,),
            )
            return cursor.lastrowid

    # ── Chat logging ────────────────────────────────────────────

    def log_chat(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[str] | None = None,
        safety_filters_fired: int = 0,
    ) -> int:
        """Log a chat message to the audit trail."""
        if not AUDIT_ENABLED:
            return -1

        now = datetime.now(timezone.utc).isoformat()
        sources_json = json.dumps(sources or [])

        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO chat_logs
                   (session_id, timestamp, role, content, sources, safety_filters_fired)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, now, role, content, sources_json, safety_filters_fired),
            )
            conn.execute(
                "UPDATE sessions SET total_chats = total_chats + 1 WHERE id = ?",
                (session_id,),
            )
            return cursor.lastrowid

    # ── Query methods ───────────────────────────────────────────

    def get_prediction_history(
        self, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get paginated prediction history, newest first."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT p.*, s.start_time as session_start
                   FROM predictions p
                   JOIN sessions s ON p.session_id = s.id
                   ORDER BY p.timestamp DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_session_log(self, session_id: str) -> dict[str, Any]:
        """Get full audit trail for a single session."""
        with self._get_conn() as conn:
            session = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            predictions = conn.execute(
                "SELECT * FROM predictions WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
            chats = conn.execute(
                "SELECT * FROM chat_logs WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
        return {
            "session": dict(session) if session else {},
            "predictions": [dict(r) for r in predictions],
            "chats": [dict(r) for r in chats],
        }

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics across all sessions."""
        with self._get_conn() as conn:
            total_preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            total_chats = conn.execute("SELECT COUNT(*) FROM chat_logs").fetchone()[0]
            total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            avg_prob = conn.execute(
                "SELECT AVG(probability) FROM predictions"
            ).fetchone()[0]
            risk_dist = conn.execute(
                "SELECT risk_category, COUNT(*) as count FROM predictions GROUP BY risk_category"
            ).fetchall()
        return {
            "total_predictions": total_preds,
            "total_chats": total_chats,
            "total_sessions": total_sessions,
            "avg_probability": round(avg_prob, 4) if avg_prob else 0,
            "risk_distribution": {r["risk_category"]: r["count"] for r in risk_dist},
        }

    def export_predictions_csv(self) -> str:
        """Export all predictions as CSV string."""
        import csv
        import io

        rows = self.get_prediction_history(limit=10000)
        if not rows:
            return ""

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()
