from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd
from CRISPRfes import datasets, features, modeling

ROOT = Path(__file__).resolve().parents[1]


class CommandTests(unittest.TestCase):
    def run_cli(self, directory, script, *args, success=True):
        result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                cwd=directory, text=True, capture_output=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_pretrained_models(self):
        frame = datasets.read_table(ROOT / "examples/example_input.csv")
        spacers, pams = datasets.split_sequences(frame, "sgRNA")
        cached = {}
        models = sorted((ROOT / "models").glob("*.joblib"))
        self.assertEqual(len(models), 23)
        for path in models:
            with self.subTest(model=path.name):
                bundle = modeling.load_bundle(path)
                key = tuple(bundle["modules"])
                if key not in cached:
                    cached[key] = features.extract_features(spacers, pams, bundle["modules"])[:2]
                X, names = cached[key]
                cols = bundle["selected_columns"]
                self.assertEqual([names[i] for i in cols], bundle["feature_names"])
                scores = modeling.decision(bundle["task"], bundle["model"], bundle["scaler"].transform(X[:, cols]))
                self.assertTrue(np.isfinite(scores).all())
                meta = json.loads(path.with_suffix(".json").read_text())
                self.assertEqual(meta["format"], "CRISPRfes-model-v1")
                self.assertEqual(meta["n_selected_features"], len(cols))

    def test_predict_batches_paths_and_scale(self):
        model = ROOT / "models/CRISPRfes_zebrafish_G3.joblib"
        with tempfile.TemporaryDirectory(prefix="CRISPRfes space ") as directory:
            work = Path(directory)
            input_csv = ROOT / "examples/example_input.csv"
            for size in [1, 256]:
                self.run_cli(work, "predict.py", "--model", model, "--input", input_csv,
                             "--batch-size", size, "--out", work / f"new folder/{size}.csv")
            first = pd.read_csv(work / "new folder/1.csv")
            other = pd.read_csv(work / "new folder/256.csv")
            np.testing.assert_allclose(first.CRISPRfes_score, other.CRISPRfes_score, rtol=1e-12, atol=1e-12)
            self.assertEqual(list(first.sgRNA), list(pd.read_csv(input_csv).sgRNA))
            self.run_cli(work, "predict.py", "--model", model, "--input", input_csv,
                         "--original-scale", "--out", work / "raw.csv")
            norm = modeling.load_bundle(model)["label_norm"]
            np.testing.assert_allclose(pd.read_csv(work / "raw.csv").CRISPRfes_score,
                                       first.CRISPRfes_score * norm["std"] + norm["mean"])
            from openpyxl import Workbook
            book = Workbook(); sheet = book.active; sheet.append(["guide", "PAM"])
            for seq in first.sgRNA:
                sheet.append([seq[:20], seq[20:]])
            book.save(work / "input.xlsx")
            self.run_cli(work, "predict.py", "--model", model, "--input", work / "input.xlsx",
                         "--seq-col", "guide", "--pam-col", "PAM", "--out", work / "xlsx.csv")
            np.testing.assert_allclose(pd.read_csv(work / "xlsx.csv").CRISPRfes_score, first.CRISPRfes_score)
            self.run_cli(work, "predict.py", "--model", model, "--input", input_csv, "--out", input_csv, success=False)

    def test_training_and_evaluation_entrypoints(self):
        rng = np.random.default_rng(123)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            frame = pd.DataFrame({
                "sgRNA": ["".join(rng.choice(list("ACGT"), 20)) + "AGG" for _ in range(36)],
                "label": [0, 1] * 18,
                "cell": np.repeat(["A", "B", "C"], 12),
            })
            frame.to_csv(work / "train.csv", index=False)
            for task in ["classification", "regression"]:
                for group in ["G1", "G2", "G3"]:
                    out = work / f"{task}_{group}.joblib"
                    self.run_cli(work, "train.py", "--data", work / "train.csv", "--task", task,
                                 "--group", group, "--B", 2, "--tau", .5, "--out", out)
                    self.run_cli(work, "predict.py", "--model", out, "--input", work / "train.csv", "--out", work / "scores.csv")
            self.run_cli(work, "evaluate.py", "--data", work / "train.csv", "--task", "classification",
                         "--protocol", "loco", "--group-col", "cell", "--folds", 2,
                         "--group", "G1", "--out", work / "loco.csv")
            result = pd.read_csv(work / "loco.csv")
            self.assertEqual(len(result), 6)
            self.assertEqual(set(result.held_out), {"A", "B", "C"})
            self.assertTrue((result.groupby("held_out").mean_pairwise_jaccard.nunique() == 1).all())
            self.run_cli(work, "reproduce_three_groups.py", "--data", work / "train.csv", "--task", "regression",
                         "--folds", 2, "--B", 2, "--tau", .5, "--zscore", "--outdir", work / "all_groups")
            summary = pd.read_csv(work / "all_groups/summary_regression_kfold.csv")
            self.assertEqual(list(summary.group), ["G1", "G2", "G3"])
            self.assertTrue(np.isfinite(summary["mean"]).all())
            for group in summary.group:
                folds = pd.read_csv(work / f"all_groups/folds_regression_kfold_{group}.csv")
                self.assertAlmostEqual(summary.loc[summary.group == group, "sd"].iloc[0], folds.Spearman.std(ddof=1))

    def test_legacy_format(self):
        bundle = modeling.load_bundle(ROOT / "models/CRISPRfes_zebrafish_G3.joblib")
        bundle["format"] = "SgFES-model-v1"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.joblib"
            joblib.dump(bundle, path)
            self.assertEqual(modeling.load_bundle(path)["format"], "CRISPRfes-model-v1")
