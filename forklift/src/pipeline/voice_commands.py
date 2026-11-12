"""Configurable Voice-to-Command & Implied Sentiment Router using Qdrant Edge Memory.

Maps open-ended operator speech (Whisper-Tiny transcript) to configurable forklift
commands and implied sentiment using few-shot sample utterances stored in
`forklift_audio_lane` (`ephemeral=False`). No SLM is required.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from src.memory.orchestrator import SiMaForkliftEdgeMemory
from src.pipeline.encoders import AudioEncoder


class CommandDefinition(BaseModel):
    command_id: str
    description: str
    implied_sentiment: str  # e.g. "urgent_hazard", "maintenance_concern", "routine_completion"
    priority: str = "NORMAL"  # "CRITICAL" | "HIGH" | "NORMAL"
    sample_utterances: List[str] = Field(default_factory=list)


class CommandInferenceResult(BaseModel):
    transcript: str
    status: str  # "MATCHED" | "AMBIGUOUS" | "NO_MATCH"
    command_id: Optional[str] = None
    implied_sentiment: Optional[str] = None
    priority: Optional[str] = None
    confidence: float = 0.0
    matched_sample: Optional[str] = None


DEFAULT_COMMANDS: List[CommandDefinition] = [
    CommandDefinition(
        command_id="EMERGENCY_STOP",
        description="Halt forklift motion immediately due to imminent collision or hazard",
        implied_sentiment="urgent_hazard",
        priority="CRITICAL",
        sample_utterances=[
            "Stop right now!",
            "Whoa, hit the brakes!",
            "Watch out, someone is stepping into the aisle!",
            "About to hit the racking, freeze!",
        ],
    ),
    CommandDefinition(
        command_id="REPORT_OBSTRUCTION",
        description="Log a blocked aisle, damaged pallet, or floor spill to WES",
        implied_sentiment="frustrated_blocked",
        priority="HIGH",
        sample_utterances=[
            "Broken pallet blocking bay 12",
            "I cannot get through this aisle, something is in the way",
            "There is debris and shrink wrap all over the floor here",
            "Aisle is jammed with a dropped load",
        ],
    ),
    CommandDefinition(
        command_id="REQUEST_MAINTENANCE",
        description="Flag mechanical, hydraulic, or battery anomaly for service",
        implied_sentiment="maintenance_concern",
        priority="HIGH",
        sample_utterances=[
            "The hydraulic pump is whining when I lift above four meters",
            "Mast feels sluggish and makes a grinding noise",
            "Steering feels heavy and battery is draining fast",
            "Forks are vibrating weirdly under load",
        ],
    ),
    CommandDefinition(
        command_id="CONFIRM_TASK_COMPLETE",
        description="Mark current WES putaway or pick order as finished",
        implied_sentiment="positive_completion",
        priority="NORMAL",
        sample_utterances=[
            "Load is safely in the rack, all done here",
            "Finished putting away this pallet, ready for the next job",
            "Bay is clear and task is wrapped up",
        ],
    ),
]


class VoiceCommandRouter:
    """Routes open-ended voice transcripts to configurable commands via Qdrant Edge."""

    def __init__(
        self,
        memory: SiMaForkliftEdgeMemory,
        audio_encoder: AudioEncoder,
        match_threshold: float = 0.55,
        ambiguous_floor: float = 0.30,
    ):
        self.memory = memory
        self.audio_encoder = audio_encoder
        self.match_threshold = match_threshold
        self.ambiguous_floor = ambiguous_floor
        self._commands: List[CommandDefinition] = []

    def load_commands_from_json(self, json_path: Path) -> int:
        raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
        cmds = [CommandDefinition(**item) for item in raw]
        return self.register_commands(cmds)

    def register_commands(self, commands: List[CommandDefinition]) -> int:
        """Embed all sample utterances into forklift_audio_lane (ephemeral=False)."""
        self._commands = list(commands)
        count = 0
        for cmd in self._commands:
            for sample in cmd.sample_utterances:
                vec = self.audio_encoder.encode_transcript(sample)
                pid = str(uuid.uuid4())
                self.memory.store.upsert(
                    SiMaForkliftEdgeMemory.AUDIO_LANE,
                    pid,
                    vec,
                    {
                        "lane": "audio",
                        "source_type": "command_sample",
                        "command_id": cmd.command_id,
                        "implied_sentiment": cmd.implied_sentiment,
                        "priority": cmd.priority,
                        "sample_text": sample,
                        "ephemeral": False,
                    },
                )
                count += 1
        self.memory.store.flush()
        return count

    def infer_command(self, raw_audio_or_transcript: bytes | str) -> CommandInferenceResult:
        """Transcribe and infer the closest configurable command & implied sentiment."""
        transcript = self.audio_encoder.transcribe(raw_audio_or_transcript)
        vec = self.audio_encoder.encode_transcript(transcript)

        hits = self.memory.query_lane(
            SiMaForkliftEdgeMemory.AUDIO_LANE,
            vec,
            top_k=1,
            score_threshold=self.ambiguous_floor,
            must_match={"source_type": "command_sample"},
        )

        if hits and hits[0].score >= self.match_threshold:
            top = hits[0]
            return CommandInferenceResult(
                transcript=transcript,
                status="MATCHED",
                command_id=top.payload["command_id"],
                implied_sentiment=top.payload["implied_sentiment"],
                priority=top.payload["priority"],
                confidence=top.score,
                matched_sample=top.payload.get("sample_text"),
            )

        if hits:
            top = hits[0]
            return CommandInferenceResult(
                transcript=transcript,
                status="AMBIGUOUS",
                command_id=top.payload["command_id"],
                implied_sentiment=top.payload["implied_sentiment"],
                priority=top.payload["priority"],
                confidence=top.score,
                matched_sample=top.payload.get("sample_text"),
            )

        return CommandInferenceResult(transcript=transcript, status="NO_MATCH")
