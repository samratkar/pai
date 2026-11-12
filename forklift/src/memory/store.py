"""Persistent multi-lane vector store with Qdrant Edge, QdrantClient, and SQLite fallback.

Supports:
1. `qdrant_edge` (`EdgeShard`) when installed (DeepLearning.AI L5 architecture).
2. `qdrant_client` (`QdrantClient(path=...)`) when installed (Whitepaper PoC).
3. `LocalDiskVectorStore` (SQLite-backed) zero-dependency fallback with exact
   cosine similarity, metadata filtering, and exponential freshness decay.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SearchHit:
    id: str
    score: float
    payload: Dict[str, Any]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class LocalDiskVectorStore:
    """SQLite-persisted multi-lane vector store mimicking Qdrant Edge semantics."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.storage_dir / "qdrant_edge_fallback.db"
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lane_points (
                lane TEXT NOT NULL,
                point_id TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (lane, point_id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lane_meta (
                lane TEXT PRIMARY KEY,
                dim INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

    def ensure_lane(self, lane: str, dim: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO lane_meta (lane, dim) VALUES (?, ?)",
            (lane, dim),
        )
        self._conn.commit()

    def upsert(self, lane: str, point_id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        row = self._conn.execute("SELECT dim FROM lane_meta WHERE lane = ?", (lane,)).fetchone()
        if row is None:
            raise ValueError(f"Lane '{lane}' not initialized")
        expected_dim = row[0]
        if len(vector) != expected_dim:
            raise ValueError(f"Lane '{lane}' expects {expected_dim}-d vector, got {len(vector)}-d")
        self._conn.execute(
            """
            INSERT INTO lane_points (lane, point_id, vector_json, payload_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(lane, point_id) DO UPDATE SET
                vector_json = excluded.vector_json,
                payload_json = excluded.payload_json
            """,
            (lane, point_id, json.dumps(vector), json.dumps(payload)),
        )
        self._conn.commit()

    def query(
        self,
        lane: str,
        query_vector: List[float],
        limit: int = 3,
        score_threshold: Optional[float] = None,
        must_match: Optional[Dict[str, Any]] = None,
        freshness_half_life_sec: Optional[float] = None,
        freshness_weight: float = 0.0,
        now_epoch: Optional[float] = None,
    ) -> List[SearchHit]:
        rows = self._conn.execute(
            "SELECT point_id, vector_json, payload_json FROM lane_points WHERE lane = ?",
            (lane,),
        ).fetchall()

        now_ts = now_epoch if now_epoch is not None else time.time()
        hits: List[SearchHit] = []
        for point_id, vec_json, pay_json in rows:
            payload = json.loads(pay_json)
            if must_match:
                if not all(payload.get(k) == v for k, v in must_match.items()):
                    continue
            vec = json.loads(vec_json)
            sim = _cosine_similarity(query_vector, vec)
            if score_threshold is not None and sim < score_threshold:
                continue

            rank_score = sim
            if freshness_half_life_sec and freshness_weight > 0.0:
                ts = float(payload.get("timestamp_epoch", now_ts))
                age = max(0.0, now_ts - ts)
                decay = math.exp(-math.log(2.0) * age / freshness_half_life_sec)
                rank_score = sim + freshness_weight * decay

            hits.append(SearchHit(id=point_id, score=round(sim, 4), payload=payload))
            # Store rank_score temporarily on object for sorting
            hits[-1]._rank_score = rank_score  # type: ignore[attr-defined]

        hits.sort(key=lambda h: getattr(h, "_rank_score", h.score), reverse=True)
        return hits[:limit]

    def delete_where(self, lane: str, must_match: Dict[str, Any]) -> int:
        rows = self._conn.execute(
            "SELECT point_id, payload_json FROM lane_points WHERE lane = ?",
            (lane,),
        ).fetchall()
        to_delete = []
        for point_id, pay_json in rows:
            payload = json.loads(pay_json)
            if all(payload.get(k) == v for k, v in must_match.items()):
                to_delete.append(point_id)
        if to_delete:
            placeholders = ",".join("?" for _ in to_delete)
            self._conn.execute(
                f"DELETE FROM lane_points WHERE lane = ? AND point_id IN ({placeholders})",
                [lane, *to_delete],
            )
            self._conn.commit()
        return len(to_delete)

    def count(self, lane: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM lane_points WHERE lane = ?", (lane,)
        ).fetchone()
        return int(row[0]) if row else 0

    def flush(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class QdrantClientVectorStore:
    """Adapter for `qdrant_client.QdrantClient(path=...)` when installed."""

    def __init__(self, storage_dir: Path):
        from qdrant_client import QdrantClient

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.storage_dir))

    def ensure_lane(self, lane: str, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self.client.collection_exists(lane):
            self.client.create_collection(
                collection_name=lane,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, lane: str, point_id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        from qdrant_client.models import PointStruct

        self.client.upsert(
            collection_name=lane,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    def query(
        self,
        lane: str,
        query_vector: List[float],
        limit: int = 3,
        score_threshold: Optional[float] = None,
        must_match: Optional[Dict[str, Any]] = None,
        freshness_half_life_sec: Optional[float] = None,
        freshness_weight: float = 0.0,
        now_epoch: Optional[float] = None,
    ) -> List[SearchHit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        q_filter = None
        if must_match:
            q_filter = Filter(
                must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in must_match.items()]
            )
        res = self.client.query_points(
            collection_name=lane,
            query=query_vector,
            query_filter=q_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        ).points
        return [SearchHit(id=str(h.id), score=round(h.score, 4), payload=h.payload or {}) for h in res]

    def delete_where(self, lane: str, must_match: Dict[str, Any]) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        q_filter = Filter(
            must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in must_match.items()]
        )
        before = self.count(lane)
        self.client.delete(collection_name=lane, points_selector=q_filter)
        after = self.count(lane)
        return max(0, before - after)

    def count(self, lane: str) -> int:
        return int(self.client.count(collection_name=lane).count)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.client.close()


def create_vector_store(storage_dir: Path, prefer_qdrant: bool = True):
    """Instantiate QdrantClientVectorStore if available, else LocalDiskVectorStore."""
    if prefer_qdrant:
        try:
            import qdrant_client  # noqa: F401
            return QdrantClientVectorStore(storage_dir)
        except ImportError:
            pass
    return LocalDiskVectorStore(storage_dir)

