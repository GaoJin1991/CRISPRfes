#!/usr/bin/env python
"""Run G1, G2, and G3 with one evaluation protocol."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run all three SgFES groups")
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--protocol", choices=["kfold", "loco"], default="kfold")
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--zscore", action="store_true")
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--outdir", default="reproduce_out")
    args, extra = parser.parse_known_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    started = time.time()
    outputs = {}
    for group in ["G1", "G2", "G3"]:
        out = os.path.join(args.outdir, f"folds_{args.task}_{args.protocol}_{group}.csv")
        command = [
            sys.executable,
            os.path.join(HERE, "evaluate.py"),
            "--data", args.data,
            "--task", args.task,
            "--protocol", args.protocol,
            "--group", group,
            "--tau", str(args.tau),
            "--out", out,
        ] + extra
        if args.group_col:
            command += ["--group-col", args.group_col]
        if args.zscore:
            command += ["--zscore"]
        print(f"\nSgFES-{group}")
        completed = subprocess.run(command, text=True, capture_output=True)
        sys.stdout.write(completed.stdout)
        if completed.returncode:
            sys.stderr.write(completed.stderr)
            return completed.returncode
        outputs[group] = out

    metric = "AUROC" if args.task == "classification" else "Spearman"
    rows = []
    for group, path in outputs.items():
        frame = pd.read_csv(path)
        rows.append({
            "group": group,
            "metric": metric,
            "mean": frame[metric].mean(),
            "sd": frame[metric].std(ddof=0),
            "mean_features": frame["n_features"].mean(),
            "mean_pairwise_jaccard": frame["mean_pairwise_jaccard"].iloc[0],
            "n_folds": len(frame),
        })
    summary = os.path.join(args.outdir, f"summary_{args.task}_{args.protocol}.csv")
    pd.DataFrame(rows).to_csv(summary, index=False)
    print(f"\nWrote {summary} ({(time.time() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
