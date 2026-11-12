"""Tests for configurable VoiceCommandRouter (pure Qdrant Edge few-shot memory)."""

from pathlib import Path

from src.config import EdgeNodeConfig
from src.memory.orchestrator import SiMaForkliftEdgeMemory
from src.pipeline.encoders import AudioEncoder
from src.pipeline.voice_commands import (
    CommandDefinition,
    DEFAULT_COMMANDS,
    VoiceCommandRouter,
)


def test_open_ended_voice_to_configurable_command_and_sentiment(tmp_path: Path) -> None:
    cfg = EdgeNodeConfig(storage_dir=tmp_path / "qdrant_edge", outbox_db_path=tmp_path / "outbox.db")
    mem = SiMaForkliftEdgeMemory(cfg, prefer_qdrant=False)
    enc = AudioEncoder(dim=384)
    router = VoiceCommandRouter(mem, enc, match_threshold=0.45, ambiguous_floor=0.20)
    try:
        stored = router.register_commands(DEFAULT_COMMANDS)
        assert stored == sum(len(c.sample_utterances) for c in DEFAULT_COMMANDS)

        # Open-ended phrasing that never says the command name "REPORT_OBSTRUCTION"
        res1 = router.infer_command("Cannot get through this aisle, something is blocking the bay")
        assert res1.status == "MATCHED"
        assert res1.command_id == "REPORT_OBSTRUCTION"
        assert res1.implied_sentiment == "frustrated_blocked"

        # Open-ended mechanical concern
        res2 = router.infer_command("The hydraulic pump is making a whining noise when lifting")
        assert res2.status == "MATCHED"
        assert res2.command_id == "REQUEST_MAINTENANCE"
        assert res2.implied_sentiment == "maintenance_concern"
    finally:
        mem.close()


def test_custom_configurable_command_and_ambiguous_gating(tmp_path: Path) -> None:
    cfg = EdgeNodeConfig(storage_dir=tmp_path / "qdrant_edge", outbox_db_path=tmp_path / "outbox.db")
    mem = SiMaForkliftEdgeMemory(cfg, prefer_qdrant=False)
    enc = AudioEncoder(dim=384)
    router = VoiceCommandRouter(mem, enc, match_threshold=0.45, ambiguous_floor=0.20)
    try:
        custom_cmds = [
            CommandDefinition(
                command_id="DIM_HEADLIGHTS",
                description="Lower mast spotlight brightness when passing another operator",
                implied_sentiment="courtesy_safety",
                priority="NORMAL",
                sample_utterances=["Lights are blinding me", "Turn down the mast spotlight"],
            )
        ]
        router.register_commands(custom_cmds)

        res = router.infer_command("Those mast lights are blinding me right now")
        assert res.status == "MATCHED"
        assert res.command_id == "DIM_HEADLIGHTS"
        assert res.implied_sentiment == "courtesy_safety"

        no_match = router.infer_command("Order coffee from the cafeteria")
        assert no_match.status in ("AMBIGUOUS", "NO_MATCH")
    finally:
        mem.close()
