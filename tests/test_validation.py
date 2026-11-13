import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from CRISPRfes import datasets, features, selection


class ValidationTests(unittest.TestCase):
    def test_sequence_formats_agree(self):
        spacer = "ACGT" * 5
        a = datasets.split_sequences(pd.DataFrame({"sgRNA": [spacer.lower() + "agg"]}), "sgRNA")
        b = datasets.split_sequences(pd.DataFrame({"guide": [spacer], "PAM": ["AGG"]}), "guide", "PAM")
        self.assertEqual(a, b)

    def test_reject_incomplete_or_truncated_sequences(self):
        for value in ["ACGT" * 5, "ACGT" * 5 + "AGGA", "ACGT" * 5 + "AG"]:
            with self.subTest(sequence=value), self.assertRaises(ValueError):
                datasets.split_sequences(pd.DataFrame({"sgRNA": [value]}), "sgRNA")

    def test_reject_conflicting_or_missing_pam(self):
        frame = pd.DataFrame({"sgRNA": ["ACGT" * 5 + "AGG"], "PAM": ["TGG"]})
        for column in ["PAM", "missing"]:
            with self.assertRaises(ValueError):
                datasets.split_sequences(frame, "sgRNA", column)

    def test_empty_input_and_nonfinite_labels(self):
        with self.assertRaises(ValueError):
            datasets.split_sequences(pd.DataFrame({"sgRNA": []}), "sgRNA")
        for task in ["classification", "regression"]:
            with self.assertRaises(ValueError):
                datasets.prepare_labels([0, np.inf], task)

    def test_unknown_module(self):
        with self.assertRaises(ValueError):
            features.extract_features(["ACGT" * 5], ["AGG"], ["module1", "typo"])

    def test_empty_core_does_not_select_unstable_features(self):
        with patch.object(selection, "selection_frequency", return_value=(np.array([.2, .3]), np.array([0, 1]))):
            with self.assertRaisesRegex(ValueError, "No features meet"):
                selection.stability_core(np.zeros((20, 2)), np.zeros(20), "regression", tau=.9)

    def test_invalid_selection_parameters(self):
        X = np.arange(40).reshape(20, 2)
        for tau in [0, 1.1, np.nan]:
            with self.assertRaises(ValueError):
                selection.stability_core(X, np.arange(20), "regression", tau=tau)
        for B in [0, -1]:
            with self.assertRaises(ValueError):
                selection.selection_frequency(X, np.arange(20), "regression", B=B)
