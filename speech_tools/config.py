# ── speech_tools/config.py ─────────────────────────────────────────────────────
"""
Centralised configuration loader.

All magic numbers that govern VAD gap merging, syllable fusion,
pause thresholds, etc. live in `config.yaml`.  Loading happens once
at import time so every legacy function can do:

    from speech_tools.config import CFG
    gap_th = CFG["vad"]["gap_merge_sec"]

and stay signature-compatible with the old code.
"""
from pathlib import Path
from typing import Any, Dict

import yaml

# Location of this file → project_root/speech_tools/config.py
_CFG_PATH = Path(__file__).with_suffix(".yaml")

with _CFG_PATH.open("r", encoding="utf-8") as _fp:
    _raw_cfg: Dict[str, Any] = yaml.safe_load(_fp)

# Exported immutable mapping
CFG: Dict[str, Any] = _raw_cfg


def save(updated: Dict[str, Any], path: Path | str | None = None) -> None:
    """
    Utility used by `calibrate.py` to write back an updated YAML.
    """
    out = Path(path) if path else _CFG_PATH
    with out.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(updated, fp, sort_keys=False, allow_unicode=True)
