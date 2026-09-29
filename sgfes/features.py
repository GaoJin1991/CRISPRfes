"""Feature extraction with configurable FM1-FM12 activation."""
from __future__ import annotations

import numpy as np

from . import feature_modules
from .config import MODULE_IDS


def extract_features(spacers, pams, modules=None, progress=None):
    enabled = list(MODULE_IDS) if modules is None else [m for m in MODULE_IDS if m in modules]
    if not enabled:
        raise ValueError("At least one feature module is required.")

    spacers = [str(value).strip().upper() for value in spacers]
    pams = [str(value).strip().upper() for value in pams]
    if len(spacers) != len(pams) or not spacers:
        raise ValueError("Spacer and PAM lists must be non-empty and equal in length.")
    if any(len(value) != 20 or set(value) - set("ACGT") for value in spacers):
        raise ValueError("Each spacer must contain 20 A/C/G/T bases.")
    if any(len(value) != 3 or set(value) - set("ACGTN") for value in pams):
        raise ValueError("Each PAM must contain 3 A/C/G/T/N bases.")

    full23 = [spacer + pam for spacer, pam in zip(spacers, pams)]
    names, column_modules, blocks = [], [], []
    for module in enabled:
        if progress:
            progress(f"Extracting {module.replace('module', 'FM')}")
        frame = feature_modules.extract_module(module, spacers, full23)
        names.extend(frame.columns.tolist())
        column_modules.extend([module] * frame.shape[1])
        blocks.append(frame.to_numpy(dtype=float))

    matrix = np.hstack(blocks)
    np.nan_to_num(matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, names, np.asarray(column_modules)
