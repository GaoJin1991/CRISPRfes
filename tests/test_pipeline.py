from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from sgfes import config, datasets, features, modeling, selection
from evaluate import folds_loco


ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    EXPECTED_ROWS = {
        "human_classification.csv": 14416,
        "human_regression.csv": 14416,
        "hek293_classification.csv": 2333,
        "hek293t_regression.csv": 2333,
        "zebrafish_crisprz.csv": 1173,
    }

    def test_release_tables(self):
        for name, rows in self.EXPECTED_ROWS.items():
            with self.subTest(name=name):
                frame = pd.read_csv(ROOT / "data" / name)
                self.assertEqual(len(frame), rows)
                self.assertFalse(frame.isna().any().any())
                self.assertFalse(frame.duplicated().any())
                self.assertTrue(frame["sgRNA"].str.fullmatch(r"[ACGT]{23}").all())

    def test_reported_zebrafish_values(self):
        frame = pd.read_csv(ROOT / "results" / "zebrafish_external_tools.csv")
        observed = dict(zip(frame["method"], frame["spearman"]))
        self.assertEqual(observed["SgFES-G1"], 0.412)
        self.assertEqual(observed["SgFES-G2"], 0.432)
        self.assertEqual(observed["SgFES-G3"], 0.456)

    def test_human_external_tools(self):
        classification = pd.read_csv(ROOT / "results" / "human_classification.csv")
        regression = pd.read_csv(ROOT / "results" / "human_regression.csv")
        external_classification = classification[~classification["method"].isin(["G1", "G2", "G3"])]
        external_regression = regression[~regression["method"].isin(["G1", "G2", "G3"])]
        self.assertEqual(len(external_classification), 24)
        self.assertEqual(len(external_regression), 15)
        self.assertEqual(
            float(classification.query("cell_line == 'HL60' and method == 'SSC'")["auroc_mean"].iloc[0]),
            0.811,
        )
        self.assertEqual(
            float(regression.query("cell_line == 'HL60' and method == 'CRISPRscan'")["spearman_mean"].iloc[0]),
            0.441,
        )

    def test_manuscript_tables(self):
        table1 = load_workbook(ROOT / "tables" / "main_table1.xlsx", read_only=True)
        table2 = load_workbook(ROOT / "tables" / "main_table2.xlsx", read_only=True)
        supplement = load_workbook(ROOT / "tables" / "supplementary_tables.xlsx", read_only=True)
        self.assertEqual(
            table1.sheetnames,
            ["Human classification", "Human regression", "Zebrafish regression"],
        )
        self.assertEqual(table2.sheetnames, ["Feature modules"])
        self.assertEqual(supplement.sheetnames, ["Table S1", "Table S2", "Table S3"])
        self.assertEqual(table1["Human classification"]["C6"].value, "0.894 ± 0.014")
        self.assertEqual(table1["Zebrafish regression"]["B6"].value, "0.456 ± 0.033")
        self.assertEqual(table2["Feature modules"]["A15"].value, "FM12 / Interaction features")
        self.assertEqual(supplement["Table S3"]["A10"].value, "SgFES-G3")
        self.assertTrue((ROOT / "tables" / "supplementary_tables.pdf").is_file())

    def test_vector_figures(self):
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        for number, name in enumerate(
            ["framework", "human_classification", "human_regression", "zebrafish"],
            1,
        ):
            with self.subTest(figure=number):
                path = ROOT / "figures" / f"figure{number}_{name}.svg"
                root = ET.parse(path).getroot()
                self.assertTrue(root.attrib.get("viewBox"))
                self.assertFalse(root.findall(".//svg:image", namespace))
                vector_count = sum(
                    len(root.findall(f".//svg:{element}", namespace))
                    for element in ["line", "rect", "polygon", "circle", "path", "text"]
                )
                self.assertGreater(vector_count, 50)


class FeatureTests(unittest.TestCase):
    def test_feature_dimensions(self):
        spacers = ["ACGTACGTACGTACGTACGT", "TGCATGCATGCATGCATGCA"]
        pams = ["AGG", "TGG"]
        matrix, names, modules = features.extract_features(spacers, pams)
        self.assertEqual(matrix.shape, (2, config.TOTAL_DIM))
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(np.isfinite(matrix).all())
        for module, _, dimension in config.MODULES:
            self.assertEqual(int(np.sum(modules == module)), dimension)

    def test_group_module_settings(self):
        self.assertEqual(config.GROUPS["G1"]["modules"], config.MODULE_IDS[:3])
        self.assertEqual(config.GROUPS["G2"]["modules"], config.MODULE_IDS)
        self.assertEqual(config.GROUPS["G3"]["modules"], config.MODULE_IDS)

    def test_arbitrary_module_activation(self):
        matrix, _, modules = features.extract_features(
            ["ACGTACGTACGTACGTACGT"],
            ["AGG"],
            modules=["module3", "module8"],
        )
        self.assertEqual(matrix.shape, (1, 336 + 192))
        self.assertEqual(set(modules), {"module3", "module8"})

    def test_model_round_trip(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(30, 12))
        y = np.asarray([0, 1] * 15)
        columns = selection.broad_selection(X, y, "classification")
        scaler, model = modeling.fit(X[:, columns], y, "classification")
        bundle = modeling.new_bundle(
            "classification", "G1", ["module1"], "broad", 0.9, 40,
            scaler, model, columns, [f"f{i}" for i in columns], len(y),
            extra={"n_features_total": X.shape[1]},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            modeling.save_bundle(path, bundle)
            loaded = modeling.load_bundle(path)
            expected = modeling.decision("classification", model, scaler.transform(X[:, columns]))
            observed = modeling.decision(
                "classification", loaded["model"], loaded["scaler"].transform(X[:, columns])
            )
            np.testing.assert_allclose(observed, expected)


class SplitTests(unittest.TestCase):
    def test_cross_cell_protocol(self):
        groups = np.repeat(["A", "B", "C"], 10)
        y = np.tile([0, 1], 15)
        splits = list(folds_loco(y, groups, "classification", n_splits=5, seed=1))
        self.assertEqual(len(splits), 15)
        for train, test, tag in splits:
            held_out = tag.split("_fold")[0]
            self.assertEqual(len(train), 16)
            self.assertEqual(len(test), 2)
            self.assertFalse(np.intersect1d(train, test).size)
            self.assertTrue(np.all(groups[test] == held_out))
            self.assertTrue(np.all(groups[train] != held_out))
            self.assertEqual(set(y[test]), {0, 1})


if __name__ == "__main__":
    unittest.main()
