"""Command-line interface for ASACA.

Run ``asaca --help`` or ``python -m asaca --help`` for the top-level help.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .inference import run_inference_and_seg
from .cognition.feature_extractor import batch_build_feature_file

__all__ = ["main"]


def _add_model_args(p: argparse.ArgumentParser) -> None:
    """Add optional --processor / --model arguments to a sub-parser."""
    p.add_argument(
        "--processor",
        type=str,
        default=str(Path("Models")),  # edit to your default
        help="Path or 🤗 model ID of the Wav2Vec2 processor (default: %(default)s)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=str(Path("Models")),  # edit to your default
        help="Path or 🤗 model ID of the fine-tuned acoustic model (default: %(default)s)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="asaca", description="ASACA toolkit CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ───── infer ───────────────────────────────────────────────────────────────
    p_inf = sub.add_parser("infer", help="Run inference on a single WAV/FLAC file")
    p_inf.add_argument("audio", type=Path, help="Audio file to analyse")
    p_inf.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("out"),
        help="Output directory for plots & report (default: %(default)s)",
    )
    _add_model_args(p_inf)

    # ───── features ────────────────────────────────────────────────────────────
    p_feat = sub.add_parser("features", help="Extract features for a meta file")
    p_feat.add_argument("meta", type=Path, help="CSV/TSV meta file")
    p_feat.add_argument(
        "--out",
        type=Path,
        default=Path("cognition_training/TUH_FV_Improved.xlsx"),
        help="Destination Excel/CSV with features (default: %(default)s)",
    )
    p_feat.add_argument(
        "--dict_dir",
        type=Path,
        required=True,
        help="Directory containing pronunciation dictionaries",
    )
    _add_model_args(p_feat)

    # ───── gui ─────────────────────────────────────────────────────────────────
    p_gui = sub.add_parser("gui", help="Launch the ASACA graphical interface")
    p_gui.set_defaults(cmd="gui")  # no extra args needed

    # ───── parse & dispatch ───────────────────────────────────────────────────
    args = parser.parse_args(argv)

    if args.cmd == "infer":
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

        proc = Wav2Vec2Processor.from_pretrained(args.processor)
        model = Wav2Vec2ForCTC.from_pretrained(args.model)

        text, feats, _ = run_inference_and_seg(
            str(args.audio), model, proc, plot_output_dir=str(args.out)
        )
        print(text)
        return 0

    if args.cmd == "features":
        batch_build_feature_file(
            args.meta,
            args.dict_dir,
            args.out,
            args.processor,
            args.model,
            device="cpu",
        )
        print(f"Features → {args.out}")
        return 0

    if args.cmd == "gui":
        # Lazy import avoids pulling PyQt5 when not needed
        from .gui import main as gui_main

        gui_main()
        return 0

    # Should never reach here because sub-parser requires a command
    parser.error("Unknown command")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
