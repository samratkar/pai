"""SQLite Store-and-Forward Outbox for Wi-Fi dead-zone resilience."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


class SQLiteOutbox:
    """Durable FIFO queue for outbound WES semantic events during offline intervals."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                sent_at TEXT
            )
            """
        )
        self._conn.commit()

    def enqueue(self, topic: str, payload: Dict[str, Any]) -> int:
        now_iso = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "INSERT INTO outbox (topic, payload_json, status, created_at) VALUES (?, ?, 'PENDING', ?)",
            (topic, json.dumps(payload), now_iso),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def pending_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM outbox WHERE status = 'PENDING'"
        ).fetchone()
        return int(row[0]) if row else 0

    def list_pending(self) -> List[Tuple[int, str, Dict[str, Any]]]:
        rows = self._conn.execute(
            "SELECT id, topic, payload_json FROM outbox WHERE status = 'PENDING' ORDER BY id ASC"
        ).fetchall()
        return [(row_id, topic, json.loads(pay_json)) for row_id, topic, pay_json in rows]

    def drain(self, publish_fn: Callable[[str, Dict[str, Any]], bool]) -> int:
        """Publish pending messages in FIFO order. Returns count of successfully sent messages."""
        sent = 0
        for row_id, topic, payload in self.list_pending():
            ok = False
            try:
                ok = bool(publish_fn(topic, payload))
            except Exception:
                ok = False

            if ok:
                now_iso = datetime.now(timezone.utc).isoformat()
                self._conn.execute(
                    "UPDATE outbox SET status = 'SENT', attempts = attempts + 1, sent_at = ? WHERE id = ?",
                    (now_iso, row_id),
                )
                self._conn.commit()
                sent += 1
            else:
                self._conn.execute(
                    "UPDATE outbox SET attempts = attempts + 1 WHERE id = ?",
                    (row_id,),
                )
                self._conn.commit()
                break
        return sent

    def close(self) -> None:
        self._conn.close()

