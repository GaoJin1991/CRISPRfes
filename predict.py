#!/usr/bin/env python
"""Score sgRNA sequences with a trained SgFES model."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sgfes import datasets, features, modeling


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score sequences with SgFES")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="CSV or XLSX sequence table")
    parser.add_argument("--out", default="sgfes_predictions.csv")
    parser.add_argument("--seq-col", default=None)
    parser.add_argument("--pam-col", default=None)
    args = parser.parse_args(argv)

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
    X, names, _ = features.extract_features(spacers, pams, modules=bundle["modules"], progress=print)

    if len(names) != bundle["n_features_total"]:
        raise RuntimeError("Feature dimension differs from the training run.")
    columns = np.asarray(bundle["selected_columns"])
    if [names[i] for i in columns] != list(bundle["feature_names"]):
        raise RuntimeError("Feature names differ from the training run.")

    scaled = bundle["scaler"].transform(X[:, columns])
    frame["sgfes_score"] = modeling.decision(bundle["task"], bundle["model"], scaled)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
