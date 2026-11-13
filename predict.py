#!/usr/bin/env python
"""Score sgRNA sequences with a trained CRISPRfes model."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CRISPRfes import datasets, features, modeling


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score sequences with CRISPRfes")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="CSV or XLSX sequence table")
    parser.add_argument("--out", default="CRISPRfes_predictions.csv")
    parser.add_argument("--seq-col", default=None)
    parser.add_argument("--pam-col", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--original-scale", action="store_true",
                        help="Undo training-label z-scoring for regression predictions")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if Path(args.out).resolve() in {Path(args.input).resolve(), Path(args.model).resolve()}:
        parser.error("The output must not overwrite the input or model.")

    bundle = modeling.load_bundle(args.model)
    frame = datasets.read_table(args.input)
    seq_col = args.seq_col or datasets.guess_column(
        frame.columns, datasets.SEQ_CANDIDATES, "sequence"
    )
    pam_col = args.pam_col or next(
        (c for c in frame.columns if str(c).lower().strip() in datasets.PAM_CANDIDATES),
        None,
    )
    spacers, pams = datasets.split_sequences(frame, seq_col, pam_col)
    columns = np.asarray(bundle["selected_columns"])
    scores = []
    for start in range(0, len(spacers), args.batch_size):
        stop = start + args.batch_size
        X, names, _ = features.extract_features(
            spacers[start:stop], pams[start:stop], modules=bundle["modules"]
        )
        if len(names) != bundle["n_features_total"]:
            raise RuntimeError("Feature dimension differs from the training run.")
        if [names[i] for i in columns] != list(bundle["feature_names"]):
            raise RuntimeError("Feature names differ from the training run.")
        scaled = bundle["scaler"].transform(X[:, columns])
        scores.extend(modeling.decision(bundle["task"], bundle["model"], scaled))
    values = np.asarray(scores)
    if args.original_scale:
        if bundle["task"] != "regression":
            parser.error("--original-scale is only available for regression models.")
        if bundle.get("label_zscored"):
            norm = bundle.get("label_norm")
            if not norm or not np.isfinite([norm.get("mean", np.nan), norm.get("std", np.nan)]).all() or norm["std"] <= 0:
                parser.error("The model has no valid training-label normalization parameters.")
            values = values * norm["std"] + norm["mean"]
    frame["CRISPRfes_score"] = values
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
