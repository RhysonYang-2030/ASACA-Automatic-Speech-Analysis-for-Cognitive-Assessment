"""Command-line interface for ASACA."""

import argparse
from pathlib import Path

from .inference import run_inference_and_seg
from .cognition.feature_extractor import batch_build_feature_file


__all__ = ["main"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="asaca")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inf = sub.add_parser("infer", help="Run inference on a WAV file")
    p_inf.add_argument("audio", type=Path)
    p_inf.add_argument("-o", "--out", type=Path, default=Path("out"))

    p_feat = sub.add_parser("features", help="Extract features from metadata")
    p_feat.add_argument("meta", type=Path)
    p_feat.add_argument("--out", type=Path, default=Path("features.xlsx"))
    p_feat.add_argument("--dict_dir", type=Path, required=True)
    p_feat.add_argument("--processor", type=str, required=True)
    p_feat.add_argument("--model", type=str, required=True)

    args = parser.parse_args(argv)

    if args.cmd == "infer":
        processor = args.processor if hasattr(args, "processor") else "epoch 27"
        model = args.model if hasattr(args, "model") else "epoch 27"
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

        proc = Wav2Vec2Processor.from_pretrained(processor)
        m = Wav2Vec2ForCTC.from_pretrained(model)
        text, feats, _ = run_inference_and_seg(
            str(args.audio), m, proc, plot_output_dir=str(args.out)
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
