"""Embedded and stability-based feature selection."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

from .config import N_EST, SEED, SUBSAMPLE_RATIO, TAU_DEFAULT, default_B


def _et(task, random_state=SEED):
    if task not in {"classification", "regression"}:
        raise ValueError(f"Unknown task: {task!r}")
    estimator = ExtraTreesClassifier if task == "classification" else ExtraTreesRegressor
    return estimator(
        n_estimators=N_EST,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )


def _varying_columns(X):
    return np.where(X.std(axis=0) > 0)[0]


def broad_selection(X, y, task, seed=SEED):
    varying = _varying_columns(X)
    if not len(varying):
        raise ValueError("No varying feature columns were found.")
    selector = SelectFromModel(_et(task, random_state=seed), threshold="mean").fit(X[:, varying], y)
    return varying[selector.get_support(indices=True)]


def selection_frequency(X, y, task, B=None, seed=SEED):
    B = default_B(task) if B is None else B
    if not isinstance(B, (int, np.integer)) or B < 1:
        raise ValueError("B must be a positive integer.")
    varying = _varying_columns(X)
    if not len(varying):
        raise ValueError("No varying feature columns were found.")
    values = X[:, varying]
    sample_size = min(len(values), max(10, int(SUBSAMPLE_RATIO * len(values))))
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(varying))
    for run in range(B):
        sample = rng.choice(len(values), size=sample_size, replace=False)
        importance = _et(task, random_state=run).fit(values[sample], y[sample]).feature_importances_
        counts += importance >= importance.mean()
    return counts / B, varying


def stability_core(X, y, task, tau=TAU_DEFAULT, B=None, seed=SEED, return_freq=False):
    if not np.isfinite(tau) or not 0 < tau <= 1:
        raise ValueError("tau must be greater than 0 and at most 1.")
    frequency, varying = selection_frequency(X, y, task, B=B, seed=seed)
    keep = np.where(frequency >= tau)[0]
    if not len(keep):
        raise ValueError("No features meet tau. Use a lower threshold or inspect the training data.")
    columns = varying[keep]
    return (columns, frequency, varying) if return_freq else columns


def select(X, y, task, scheme, tau=TAU_DEFAULT, B=None, seed=SEED):
    if scheme == "broad":
        return broad_selection(X, y, task, seed=seed)
    if scheme == "stability":
        return stability_core(X, y, task, tau=tau, B=B, seed=seed)
    raise ValueError(f"Unknown selection scheme: {scheme!r}")


def mean_pairwise_jaccard(feature_sets):
    sets = [set(values) for values in feature_sets]
    scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            scores.append(len(sets[i] & sets[j]) / len(union) if union else 0.0)
    return float(np.mean(scores)) if scores else float("nan")
