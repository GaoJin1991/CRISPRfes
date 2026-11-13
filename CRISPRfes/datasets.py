"""Input-table parsing and sequence validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

SEQ_CANDIDATES = [
    "sgrna", "target seq clean", "target_seq", "target seq", "sequence",
    "spacer", "guide", "seq", "seq23",
]
PAM_CANDIDATES = ["pam"]
LABEL_CANDIDATES = [
    "label", "normalized efficacy", "normalized_efficacy",
    "somatic activity numeric", "somatic_activity", "efficacy", "activity",
    "score", "y", "target",
]


def read_table(path):
    if str(path).lower().endswith(".xlsx"):
        return pd.read_excel(path)
    if str(path).lower().endswith(".xls"):
        raise ValueError("Save legacy .xls input as .xlsx or .csv first.")
    return pd.read_csv(path)


def guess_column(columns, candidates, what):
    lower = {str(c).lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    for candidate in candidates:
        for name, original in lower.items():
            if candidate in name:
                return original
    raise ValueError(f"Could not identify the {what} column. Available columns: {list(columns)}")


def split_sequences(df, seq_col, pam_col=None):
    """Return 20-nt spacers and 3-nt PAMs."""
    if df.empty:
        raise ValueError("The input table contains no sequences.")
    if seq_col not in df.columns:
        raise ValueError(f"Sequence column {seq_col!r} was not found.")
    if df[seq_col].isna().any():
        raise ValueError("The sequence column contains missing values.")
    seqs = df[seq_col].astype(str).str.strip().str.upper()
    lengths = seqs.str.len()
    if not lengths.isin([20, 23]).all():
        raise ValueError("Each sequence must be exactly 20 or 23 nt; sequences are not truncated.")
    if pam_col is not None:
        if pam_col not in df.columns:
            raise ValueError(f"PAM column {pam_col!r} was not found.")
        if df[pam_col].isna().any():
            raise ValueError("The PAM column contains missing values.")
        spacers = seqs.str[:20]
        pams = df[pam_col].astype(str).str.strip().str.upper()
        embedded = lengths.eq(23)
        if not seqs[embedded].str[20:].eq(pams[embedded]).all():
            raise ValueError("The PAM column conflicts with the PAM in a 23-nt sequence.")
    else:
        if not lengths.eq(23).all():
            raise ValueError("20-nt spacers require a PAM column. Supply the actual 3-nt PAM.")
        spacers, pams = seqs.str[:20], seqs.str[20:23]

    if not spacers.str.fullmatch(r"[ACGT]{20}").all():
        raise ValueError("Spacer sequences must contain exactly 20 A/C/G/T bases.")
    if not pams.str.fullmatch(r"[ACGTN]{3}").all():
        raise ValueError("PAM sequences must contain exactly 3 A/C/G/T/N bases.")
    return spacers.tolist(), pams.tolist()


def _is_num(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def prepare_labels(values, task):
    y = pd.Series(values)
    if y.isna().any():
        raise ValueError("The label column contains missing values.")
    if task == "regression":
        try:
            values = y.astype(float).values
        except (TypeError, ValueError) as exc:
            raise ValueError("Regression labels must be numeric.") from exc
        if not np.isfinite(values).all():
            raise ValueError("Regression labels must be finite.")
        return values, None
    if task != "classification":
        raise ValueError(f"Unknown task: {task!r}")
    unique = list(pd.unique(y))
    if len(unique) != 2:
        raise ValueError(f"Classification requires two label values; found {len(unique)}.")
    numeric = all(_is_num(value) for value in unique)
    if numeric and not all(np.isfinite(float(value)) for value in unique):
        raise ValueError("Classification labels must be finite.")
    order = sorted(unique, key=float if numeric else str)
    mapping = {order[0]: 0, order[1]: 1}
    return y.map(mapping).astype(int).values, mapping


def load_dataset(path, task, seq_col=None, pam_col=None, label_col=None, group_col=None):
    df = read_table(path)
    seq_col = seq_col or guess_column(df.columns, SEQ_CANDIDATES, "sequence")
    label_col = label_col or guess_column(df.columns, LABEL_CANDIDATES, "label")
    if pam_col is None:
        pam_col = next(
            (c for c in df.columns if str(c).lower().strip() in PAM_CANDIDATES),
            None,
        )
    if group_col and group_col not in df.columns:
        raise ValueError(f"Group column {group_col!r} was not found.")
    if group_col and df[group_col].isna().any():
        raise ValueError("The group column contains missing values.")
    spacers, pams = split_sequences(df, seq_col, pam_col)
    y, mapping = prepare_labels(df[label_col].values, task)
    groups = df[group_col].astype(str).values if group_col else None
    return {
        "frame": df,
        "spacers": spacers,
        "pams": pams,
        "y": np.asarray(y),
        "groups": groups,
        "label_mapping": mapping,
        "columns": {
            "sequence": seq_col,
            "pam": pam_col,
            "label": label_col,
            "group": group_col,
        },
    }


def zscore_within(y, index=None):
    y = np.asarray(y, dtype=float)
    subset = y if index is None else y[index]
    mean, std = float(np.mean(subset)), float(np.std(subset))
    return (y - mean) / (std if std > 0 else 1.0)
