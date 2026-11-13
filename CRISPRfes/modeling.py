"""SVM fitting, scoring, and model serialization."""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from .config import SEED, SVM_C, SVM_GAMMA


def make_model(task):
    if task == "classification":
        return SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA, random_state=SEED)
    if task == "regression":
        return SVR(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA)
    raise ValueError(f"Unknown task: {task!r}")


def fit(X, y, task):
    scaler = StandardScaler().fit(X)
    model = make_model(task).fit(scaler.transform(X), y)
    return scaler, model


def decision(task, model, scaled_X):
    return model.decision_function(scaled_X) if task == "classification" else model.predict(scaled_X)


def score(task, y_true, y_pred):
    if task == "classification":
        return float(roc_auc_score(y_true, y_pred))
    if task == "regression":
        return float(spearmanr(y_true, y_pred).correlation)
    raise ValueError(f"Unknown task: {task!r}")


def save_bundle(path, bundle):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    joblib.dump(bundle, path, compress=("xz", 6))
    meta = {
        key: value for key, value in bundle.items()
        if key not in {"model", "scaler", "selected_columns", "feature_names"}
    }
    meta["n_selected_features"] = len(bundle["selected_columns"])
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return path


def load_bundle(path):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("format") not in {
        "CRISPRfes-model-v1", "CrisprFes-model-v1", "SgFES-model-v1",
    }:
        raise ValueError("Unsupported model bundle format.")
    required = {"task", "modules", "model", "scaler", "selected_columns",
                "feature_names", "n_features_total"}
    missing = required - bundle.keys()
    if missing:
        raise ValueError(f"Model bundle is missing fields: {sorted(missing)}")
    if bundle["task"] not in {"classification", "regression"}:
        raise ValueError("Invalid model task.")
    columns = np.asarray(bundle["selected_columns"])
    if (columns.ndim != 1 or not len(columns)
            or not np.issubdtype(columns.dtype, np.integer)
            or len(np.unique(columns)) != len(columns)
            or np.any(columns < 0) or np.any(columns >= bundle["n_features_total"])
            or len(bundle["feature_names"]) != len(columns)):
        raise ValueError("Invalid selected-feature mapping in the model bundle.")
    if any(obj.n_features_in_ != len(columns) for obj in (bundle["model"], bundle["scaler"])):
        raise ValueError("Model and scaler dimensions do not match the selected features.")
    bundle["format"] = "CRISPRfes-model-v1"
    return bundle


def new_bundle(task, group, modules, scheme, tau, B, scaler, model,
               selected_columns, feature_names, n_train, extra=None):
    bundle = {
        "format": "CRISPRfes-model-v1",
        "task": task,
        "group": group,
        "modules": list(modules),
        "selection": scheme,
        "tau": tau if scheme == "stability" else None,
        "B": B if scheme == "stability" else None,
        "svm": {"kernel": "rbf", "C": SVM_C, "gamma": SVM_GAMMA},
        "n_train": int(n_train),
        "selected_columns": np.asarray(selected_columns),
        "feature_names": list(feature_names),
        "scaler": scaler,
        "model": model,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        bundle.update(extra)
    return bundle
