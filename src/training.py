"""
Train the PCA + SVM gait classifier on the TRAIN split and save a pickle for the UI.

Usage (from project root):
    python src/training.py

Or from ``src``:
    python training.py

Requires: scikit-learn, pandas, numpy (same as Classifier / SignalReader).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_model_path() -> Path:
    return _project_root() / "models" / "grf_pca_svm_classifier.pkl"


def main(argv: list[str] | None = None) -> int:
    root = _project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    parser = argparse.ArgumentParser(description="Train PCA+SVM classifier and save model.")
    parser.add_argument(
        "--grf",
        type=Path,
        default=root / "Data" / "GRF_F_V_PRO_left.csv",
        help="GRF CSV (101 waveform columns after ID columns).",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=root / "Data" / "GRF_metadata.csv",
        help="Metadata CSV with CLASS_LABEL and TRAIN/TEST columns.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=default_model_path(),
        help="Output path for the trained classifier (.pkl).",
    )
    args = parser.parse_args(argv)

    from Classifier import Classifier
    from SignalReader import SignalReader

    if not args.grf.is_file() or not args.meta.is_file():
        print("Missing input file(s).", file=sys.stderr)
        print(f"  GRF:  {args.grf} ({args.grf.is_file()})", file=sys.stderr)
        print(f"  Meta: {args.meta} ({args.meta.is_file()})", file=sys.stderr)
        return 1

    reader = SignalReader(str(args.grf), str(args.meta))
    clf = Classifier()
    clf.fit_from_reader(reader)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clf.save(args.output)

    test_metrics = clf.evaluate_on_split(reader, "TEST")
    print(f"Saved model to: {args.output}")
    print(
        f"TEST split - accuracy: {test_metrics['accuracy']:.4f}, "
        f"balanced_accuracy: {test_metrics['balanced_accuracy']:.4f}, "
        f"n={int(test_metrics['n'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
