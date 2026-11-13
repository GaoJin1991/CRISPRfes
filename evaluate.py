#!/usr/bin/env python
"""Evaluate one CRISPRfes group with the paper protocols."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CRISPRfes import config, datasets, features, modeling, selection


def folds_kfold(y, task, n_splits, seed):
    if task == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y))))


def folds_loco(y, groups, task, n_splits=5, seed=1):
    """Yield the 15 cross-cell folds used for the human experiments."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    group_names = sorted(np.unique(groups), key=lambda value: str(value).casefold())
    if len(group_names) < 2:
        raise ValueError("Cross-cell evaluation requires at least two groups.")

    by_group = {}
    for group in group_names:
        index = np.where(groups == group)[0]
        local_splits = folds_kfold(y[index], task, n_splits, seed)
        by_group[group] = [(index[train], index[test]) for train, test in local_splits]

    for held_out in group_names:
        for fold in range(n_splits):
            train = np.concatenate([
                by_group[group][fold][0] for group in group_names if group != held_out
            ])
            test = by_group[held_out][fold][1]
            yield train, test, f"{held_out}_fold{fold + 1}"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate CRISPRfes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="CSV or XLSX input")
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--protocol", choices=["kfold", "loco"], default="kfold")
    parser.add_argument("--group", choices=sorted(config.GROUPS), default=config.DEFAULT_GROUP)
    parser.add_argument("--tau", type=float, default=config.TAU_DEFAULT)
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seq-col", default=None)
    parser.add_argument("--pam-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--zscore", action="store_true")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--fold-seed", type=int, default=1)
    parser.add_argument("--out", default="")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.protocol == "loco" and not args.group_col:
        raise SystemExit("--group-col is required for the loco protocol.")

    started = time.time()
    spec = config.GROUPS[args.group]
    runs = args.B if args.B is not None else config.default_B(args.task)
    print(f"CRISPRfes-{args.group}: {spec['desc']}")

    data = datasets.load_dataset(
        args.data, args.task, args.seq_col, args.pam_col, args.label_col, args.group_col
    )
    y = data["y"].astype(float) if args.task == "regression" else data["y"]
    X, _, _ = features.extract_features(
        data["spacers"], data["pams"], modules=spec["modules"], progress=print
    )
    print(f"Samples: {len(y)}; features: {X.shape[1]}")

    if args.protocol == "loco":
        splits = list(folds_loco(y, data["groups"], args.task, args.folds, args.fold_seed))
    else:
        splits = [
            (train, test, f"fold{fold + 1}")
            for fold, (train, test) in enumerate(
                folds_kfold(y, args.task, args.folds, args.fold_seed)
            )
        ]

    metric = "AUROC" if args.task == "classification" else "Spearman"
    rows, selected_sets = [], []
    for train, test, tag in splits:
        columns = selection.select(
            X[train], y[train], args.task, spec["selection"],
            tau=args.tau, B=runs, seed=args.seed,
        )
        fit_y = y[train]
        if args.task == "regression" and args.zscore:
            mean = float(fit_y.mean())
            std = float(fit_y.std()) or 1.0
            fit_y = (fit_y - mean) / std
        scaler, model = modeling.fit(X[np.ix_(train, columns)], fit_y, args.task)
        prediction = modeling.decision(
            args.task, model, scaler.transform(X[np.ix_(test, columns)])
        )
        value = modeling.score(args.task, y[test], prediction)
        selected_sets.append(set(columns.tolist()))
        rows.append({
            "fold": tag,
            "held_out": tag.rsplit("_fold", 1)[0] if args.protocol == "loco" else "all",
            "n_train": len(train),
            "n_test": len(test),
            "n_features": len(columns),
            metric: value,
        })
        print(
            f"{tag:18s} train={len(train):5d} test={len(test):5d} "
            f"features={len(columns):5d} {metric}={value:.4f}"
        )

    values = np.asarray([row[metric] for row in rows])
    jaccards = {}
    for held_out in dict.fromkeys(row["held_out"] for row in rows):
        indices = [i for i, row in enumerate(rows) if row["held_out"] == held_out]
        jaccards[held_out] = selection.mean_pairwise_jaccard([selected_sets[i] for i in indices])
        for i in indices:
            rows[i]["mean_pairwise_jaccard"] = jaccards[held_out]
    print(f"{metric}: {values.mean():.4f} +/- {values.std(ddof=1):.4f}")
    print(f"Mean selected features: {np.mean([row['n_features'] for row in rows]):.1f}")
    for held_out, value in jaccards.items():
        print(f"Mean pairwise Jaccard ({held_out}): {value:.4f}")
    print(f"Elapsed: {(time.time() - started) / 60:.1f} min")

    if args.out:
        result = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        result.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
