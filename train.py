#!/usr/bin/env python
"""Train and save a CRISPRfes model."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CRISPRfes import config, datasets, features, modeling, selection


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train CRISPRfes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="CSV or XLSX training table")
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--group", choices=sorted(config.GROUPS), default=config.DEFAULT_GROUP)
    parser.add_argument("--tau", type=float, default=config.TAU_DEFAULT)
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--seq-col", default=None)
    parser.add_argument("--pam-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--hold-out", default=None)
    parser.add_argument("--zscore", action="store_true")
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    started = time.time()
    spec = config.GROUPS[args.group]
    runs = args.B if args.B is not None else config.default_B(args.task)

    data = datasets.load_dataset(
        args.data, args.task, args.seq_col, args.pam_col, args.label_col, args.group_col
    )
    y = data["y"]
    keep = None
    if args.hold_out:
        if data["groups"] is None:
            raise SystemExit("--group-col is required with --hold-out.")
        keep = np.where(data["groups"] != args.hold_out)[0]
        if len(keep) in {0, len(y)}:
            raise SystemExit(f"Invalid hold-out value: {args.hold_out!r}")

    X, names, _ = features.extract_features(
        data["spacers"], data["pams"], modules=spec["modules"], progress=print
    )
    if keep is not None:
        X, y = X[keep], y[keep]

    columns = selection.select(
        X, y, args.task, spec["selection"], tau=args.tau, B=runs, seed=args.seed
    )
    fit_y, label_norm = y, None
    if args.task == "regression" and args.zscore:
        mean, std = float(np.mean(y)), float(np.std(y)) or 1.0
        fit_y = (y - mean) / std
        label_norm = {"mean": mean, "std": std}

    scaler, model = modeling.fit(X[:, columns], fit_y, args.task)
    suffix = f"_holdout_{args.hold_out}" if args.hold_out else ""
    out = args.out or os.path.join("models", f"CRISPRfes_{args.task}_{args.group}{suffix}.joblib")
    bundle = modeling.new_bundle(
        args.task,
        args.group,
        spec["modules"],
        spec["selection"],
        args.tau,
        runs,
        scaler,
        model,
        columns,
        [names[i] for i in columns],
        len(y),
        extra={
            "source_data": os.path.basename(args.data),
            "held_out": args.hold_out,
            "label_zscored": bool(args.task == "regression" and args.zscore),
            "label_norm": label_norm,
            "n_features_total": int(X.shape[1]),
            "class_labels": ([str(label) for label in data["label_mapping"]]
                             if data["label_mapping"] is not None else None),
        },
    )
    modeling.save_bundle(out, bundle)
    print(
        f"Saved {out} ({len(columns)}/{X.shape[1]} features; "
        f"{(time.time() - started) / 60:.1f} min)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
