"""
SQLite-backed chat history store.

Each conversation is identified by a session_id (either a user-chosen name or
the Gradio-assigned session_hash). This enables cross-session persistence and
true multi-user support — each user's history is stored and retrieved independently.

Tables:
  messages  — every individual chat message (full audit trail)
  summaries — rolling summary of older messages per session (used for LLM context)
"""

import sqlite3
from paths import CHAT_HISTORY_DB_FPATH


class ChatHistoryDB:
    def __init__(self, db_path: str = str(CHAT_HISTORY_DB_FPATH)):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    session_id TEXT PRIMARY KEY,
                    summary    TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)"
            )

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def save_message(self, session_id: str, role: str, content: str):
        """Append a single message to the session's history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )

    def load_history(self, session_id: str) -> list[dict]:
        """Return all messages for a session in chronological order."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]

    def clear_history(self, session_id: str):
        """Delete all messages and the summary for a session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))

    # ------------------------------------------------------------------
    # Rolling summary
    # ------------------------------------------------------------------

    def get_summary(self, session_id: str) -> str | None:
        """Return the stored rolling summary, or None if none exists."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT summary FROM summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row[0] if row else None

    def save_summary(self, session_id: str, summary: str):
        """Upsert the rolling summary for a session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO summaries (session_id, summary, updated_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP)",
                (session_id, summary),
            )

    # ------------------------------------------------------------------
    # Session listing (for awareness / admin)
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[dict]:
        """Return all sessions ordered by most recently active."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT session_id, MAX(timestamp) AS last_active, COUNT(*) AS msg_count "
                "FROM messages GROUP BY session_id ORDER BY last_active DESC"
            ).fetchall()
        return [
            {"session_id": r[0], "last_active": r[1], "messages": r[2]}
            for r in rows
        ]
