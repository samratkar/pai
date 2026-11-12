"""On-device multimodal encoders (SiMa.ai Palette INT8 / FastEmbed / deterministic fallback).

In production on SiMa.ai MLSoC, these wrap `mla-rt` INT8 compiled models:
- MobileCLIP-S0 (512-d vision)
- Whisper-Tiny + bge-small-en-v1.5 (384-d audio/transcript)
- StateSerializer + bge-small-en-v1.5 (384-d CAN-bus telemetry)

For local prototyping without NPU hardware, a deterministic n-gram feature
projection produces L2-normalized unit vectors with realistic cosine geometry.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List, Optional


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _hashed_ngram_embedding(text: str, dim: int) -> List[float]:
    """Deterministic L2-normalized bag-of-tokens + char-3gram projection."""
    vec = [0.0] * dim
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if not normalized:
        vec[0] = 1.0
        return vec

    tokens = normalized.split()
    features: List[tuple[str, float]] = []
    for tok in tokens:
        features.append((f"tok:{tok}", 2.0))
        padded = f"<{tok}>"
        for i in range(max(1, len(padded) - 2)):
            features.append((f"tri:{padded[i:i+3]}", 1.0))

    for feat, weight in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "little")
        idx = h % dim
        sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
        vec[idx] += sign * weight

    return _l2_normalize(vec)


class VisionEncoder:
    """512-d MobileCLIP-S0 encoder (with optional FastEmbed CLIP backend)."""

    def __init__(self, dim: int = 512, use_fastembed: bool = False):
        self.dim = dim
        self._clip_model = None
        if use_fastembed:
            try:
                from fastembed import ImageEmbedding
                self._clip_model = ImageEmbedding("Qdrant/clip-ViT-B-32-vision")
            except Exception:
                self._clip_model = None

    def encode_label_or_frame(self, description_or_label: str) -> List[float]:
        """Encode a visual scene descriptor or ROI tag into a 512-d unit vector."""
        return _hashed_ngram_embedding(description_or_label, self.dim)

    def encode_image_path(self, image_path: str, fallback_hint: Optional[str] = None) -> List[float]:
        """Encode an image file via CLIP if available, else deterministic fallback."""
        if self._clip_model is not None:
            vecs = list(self._clip_model.embed([image_path]))
            return [float(x) for x in vecs[0]]
        hint = fallback_hint or image_path
        return _hashed_ngram_embedding(hint, self.dim)


class AudioEncoder:
    """Whisper-Tiny ASR + 384-d bge-small-en-v1.5 semantic encoder."""

    def __init__(self, dim: int = 384, use_fastembed: bool = False):
        self.dim = dim
        self._text_model = None
        if use_fastembed:
            try:
                from fastembed import TextEmbedding
                self._text_model = TextEmbedding("BAAI/bge-small-en-v1.5")
            except Exception:
                self._text_model = None

    def transcribe(self, audio_bytes_or_text: bytes | str) -> str:
        """Simulate on-device Whisper-Tiny ASR when given raw text/bytes."""
        if isinstance(audio_bytes_or_text, str):
            return audio_bytes_or_text
        return audio_bytes_or_text.decode("utf-8", errors="replace")

    def encode_transcript(self, transcript: str) -> List[float]:
        if self._text_model is not None:
            vecs = list(self._text_model.embed([transcript]))
            return [float(x) for x in vecs[0]]
        return _hashed_ngram_embedding(transcript, self.dim)


class TelemetryEncoder:
    """SocketCAN state serializer + 384-d bge-small-en-v1.5 semantic encoder."""

    def __init__(self, dim: int = 384, use_fastembed: bool = False):
        self.dim = dim
        self._text_model = None
        if use_fastembed:
            try:
                from fastembed import TextEmbedding
                self._text_model = TextEmbedding("BAAI/bge-small-en-v1.5")
            except Exception:
                self._text_model = None

    @staticmethod
    def serialize_can_state(
        action: str,
        load_weight_kg: float,
        sku_id: Optional[str],
        zone_id: str,
        mast_height_mm: int = 0,
        battery_soc_pct: float = 100.0,
    ) -> str:
        sku_part = f" {sku_id}" if sku_id else ""
        return (
            f"{action.capitalize()} {load_weight_kg:.0f}kg{sku_part} in {zone_id}; "
            f"mast {mast_height_mm}mm, battery {battery_soc_pct:.0f}%."
        )

    def encode_state(self, semantic_summary: str) -> List[float]:
        if self._text_model is not None:
            vecs = list(self._text_model.embed([semantic_summary]))
            return [float(x) for x in vecs[0]]
        return _hashed_ngram_embedding(semantic_summary, self.dim)

    def encode_batch(self, summaries: Iterable[str]) -> List[List[float]]:
        return [self.encode_state(s) for s in summaries]

