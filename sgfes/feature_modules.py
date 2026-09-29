"""Implementations of all 12 SgFES feature modules."""

import math
from itertools import product
from collections import Counter

import numpy as np
import pandas as pd

# Nucleotide constants
NTS     = list("ACGT")
SEQ_NTS = list("ACTG")
DINUCS  = [a+b for a, b in product(NTS, NTS)]          # 16
TRINUCS = [a+b+c for a, b, c in product(NTS, NTS, NTS)]  # 64
TETRAS  = [a+b+c+d for a, b, c, d in product(NTS, NTS, NTS, NTS)]  # 256


# FM1-FM3

def extract_module1(spacers, full23):
    columns = {}
    for size in range(1, 5):
        for bases in product(SEQ_NTS, repeat=size):
            motif = "".join(bases)
            columns[motif] = np.fromiter(
                (seq.count(motif) for seq in full23),
                dtype=np.int8,
                count=len(full23),
            )
    return pd.DataFrame(columns)


def extract_module2(spacers, full23):
    length = len(full23[0])
    columns = {}
    for size in range(1, 5):
        for bases in product(SEQ_NTS, repeat=size):
            motif = "".join(bases)
            for pos in range(length - size + 1):
                columns[f"{motif}_{pos + 1}"] = np.fromiter(
                    (seq[pos:pos + size] == motif for seq in full23),
                    dtype=np.int8,
                    count=len(full23),
                )
    return pd.DataFrame(columns)


def extract_module3(spacers, full23):
    length = len(full23[0])
    columns = {}
    for gap in range(1, length - 1):
        for first in SEQ_NTS:
            for second in SEQ_NTS:
                name = f"GAP_{first}_{gap}_{second}"
                columns[name] = np.fromiter(
                    (
                        sum(
                            seq[j] == first and seq[j + gap + 1] == second
                            for j in range(length - gap - 1)
                        )
                        for seq in full23
                    ),
                    dtype=np.int8,
                    count=len(full23),
                )
    return pd.DataFrame(columns)

# Sequence helpers

def kmer_counts(seq, k):
    cnt = Counter()
    for i in range(len(seq) - k + 1):
        w = seq[i:i+k]
        if all(c in "ACGT" for c in w):
            cnt[w] += 1
    return cnt


def ngd_counts_in(seq, gap):
    """Count all 16 dinuc pairs at given gap in seq.
    gap=k gives the pair (seq[j], seq[j+k+1]).
    Returns dict keyed by 2-char dinuc string."""
    cnt = {d: 0 for d in DINUCS}
    L = len(seq)
    for j in range(L - gap - 1):
        d = seq[j] + seq[j + gap + 1]
        if d in cnt:
            cnt[d] += 1
    return cnt


def shannon_entropy(seq):
    cnt = Counter(seq)
    total = len(seq)
    if total == 0:
        return 0.0
    h = 0.0
    for n in cnt.values():
        p = n / total
        h -= p * math.log2(p)
    return h


def max_homopolymer(seq):
    if not seq:
        return 0
    best = cur = 1
    for i in range(1, len(seq)):
        cur = cur + 1 if seq[i] == seq[i-1] else 1
        best = max(best, cur)
    return best


def max_run_of(seq, nt):
    best = cur = 0
    for c in seq:
        cur = cur + 1 if c == nt else 0
        best = max(best, cur)
    return best


def count_runs_of(seq, nt, min_len=2):
    count = cur = 0
    for c in seq + " ":  # sentinel
        if c == nt:
            cur += 1
        else:
            if cur >= min_len:
                count += 1
            cur = 0
    return count


def count_nonoverlap(seq, pattern):
    count = 0
    i = 0
    L = len(pattern)
    while i <= len(seq) - L:
        if seq[i:i+L] == pattern:
            count += 1
            i += L
        else:
            i += 1
    return count


def reverse_complement(seq):
    c = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(c.get(x, "N") for x in reversed(seq))


def self_comp_count(seq):
    rc = reverse_complement(seq)
    return sum(a == b for a, b in zip(seq, rc))


def max_self_comp_window(seq, k):
    rc = reverse_complement(seq)
    best = 0
    for i in range(len(seq) - k + 1):
        best = max(best, sum(a == b for a, b in zip(seq[i:i+k], rc[i:i+k])))
    return best


def wallace_tm(seq):
    return 4 * (seq.count("G") + seq.count("C")) + 2 * (seq.count("A") + seq.count("T"))


def dg_proxy(seq):
    return -(2.0 * seq.count("G") + 2.0 * seq.count("C") + 0.5 * seq.count("T"))


# Spacer k-mer composition

def _module4_row(spacer):
    f = {}
    mc = kmer_counts(spacer, 1)
    for nt in NTS:
        f[f"t2_mono_{nt}"] = mc.get(nt, 0)

    dc = kmer_counts(spacer, 2)
    for d in DINUCS:
        f[f"t2_di_{d}"] = dc.get(d, 0)

    tc = kmer_counts(spacer, 3)
    for t in TRINUCS:
        f[f"t2_tri_{t}"] = tc.get(t, 0)

    qc = kmer_counts(spacer, 4)
    for q in TETRAS:
        f[f"t2_tetra_{q}"] = qc.get(q, 0)

    L = len(spacer)
    gc = spacer.count("G") + spacer.count("C")
    at = spacer.count("A") + spacer.count("T")
    f["t2_gc_count"]       = gc
    f["t2_gc_frac"]        = gc / L if L else 0.0
    f["t2_at_frac"]        = at / L if L else 0.0
    f["t2_purine_frac"]    = (spacer.count("A") + spacer.count("G")) / L if L else 0.0
    f["t2_pyrimidine_frac"]= (spacer.count("C") + spacer.count("T")) / L if L else 0.0
    f["t2_gc_gt10"]        = int(gc > 10)
    f["t2_gc_lt10"]        = int(gc < 10)
    for k in range(1, 21):
        f[f"t2_gc_eq{k}"]  = int(gc == k)
    # Targeted dinucs (named separately; redundant with general di counts)
    f["t2_cpg"] = dc.get("CG", 0)
    f["t2_gpc"] = dc.get("GC", 0)
    f["t2_tpa"] = dc.get("TA", 0)
    return f
# Extended position-specific features

def _module5_row(spacer):
    f = {}
    L = len(spacer)
    # PSF-1mer on spacer (positions 1..20)
    for pos in range(1, L + 1):
        nt = spacer[pos - 1]
        for n in NTS:
            f[f"t3_psf1_{n}_{pos}"] = int(nt == n)
    # PSF-2mer on spacer (adjacent pairs, positions 1..19)
    for pos in range(1, L):
        di = spacer[pos-1:pos+1]
        for d in DINUCS:
            f[f"t3_psf2_{d}_{pos}"] = int(di == d)
    # Per-position GC and purine indicators
    for pos in range(1, L + 1):
        f[f"t3_gc_{pos}"]  = int(spacer[pos-1] in "GC")
        f[f"t3_pur_{pos}"] = int(spacer[pos-1] in "AG")
    # Named single-position bookend indicators
    f["t3_pos1_A"]  = int(spacer[0] == "A")  if L >= 1  else 0
    f["t3_pos20_G"] = int(spacer[19] == "G") if L >= 20 else 0
    return f


# Full and regional gapped dinucleotides

def _module6_row(spacer, full23):
    f = {}
    regions = {
        "seed":    (spacer[12:20],    range(1, 6)),   # 8-nt, gap 1-5
        "pampx":   (full23[16:23],    range(1, 6)),   # 7-nt, gap 1-5
        "5p":      (spacer[0:10],     range(1, 6)),   # 10-nt, gap 1-5
        "mid":     (spacer[6:16],     range(1, 6)),   # 10-nt, gap 1-5
        "extseed": (full23[12:23],    range(1, 11)),  # 11-nt, gap 1-10
    }
    for rname, (seq, gaps) in regions.items():
        for gap in gaps:
            cnt = ngd_counts_in(seq, gap)
            for d in DINUCS:
                f[f"t4_{rname}_ngd_{d}_{gap}"] = cnt[d]
    return f


# Regional summary features

def _region_summary(seq, prefix):
    f = {}
    L = len(seq)
    if L == 0:
        return f
    gc = seq.count("G") + seq.count("C")
    at = seq.count("A") + seq.count("T")
    f[f"{prefix}gc_count"]   = gc
    f[f"{prefix}gc_frac"]    = gc / L
    f[f"{prefix}at_count"]   = at
    f[f"{prefix}at_frac"]    = at / L
    for nt in NTS:
        f[f"{prefix}mono_{nt}"] = seq.count(nt)
    f[f"{prefix}purine_frac"] = (seq.count("A") + seq.count("G")) / L
    dc = kmer_counts(seq, 2)
    for d in DINUCS:
        f[f"{prefix}di_{d}"] = dc.get(d, 0)
    f[f"{prefix}entropy"] = shannon_entropy(seq)
    f[f"{prefix}max_hp"]  = max_homopolymer(seq)
    return f


def _module7_row(spacer, full23):
    f = {}
    seqs = {
        "seed":   spacer[12:20],
        "pampx":  full23[16:23],
        "mid":    spacer[6:16],
        "5p":     spacer[0:10],
        "fullsp": spacer,
    }
    for rname, seq in seqs.items():
        f.update(_region_summary(seq, f"t5_{rname}_"))
    return f


# PAM and junction motifs

def _module8_row(spacer, full23):
    f = {}
    pam = full23[20:23]

    for p in ["TGG", "CGG", "AGG", "GGG"]:
        f[f"t6_pam_{p}"] = int(pam == p)
    for nt in NTS:
        f[f"t6_pam0_{nt}"] = int(len(pam) >= 1 and pam[0] == nt)
    f["t6_pam1_G"] = int(len(pam) >= 2 and pam[1] == "G")

    # Dinucs at PAM[0]-PAM[1], spacer[-1]-PAM[0], PAM[1]-PAM[2]
    pam01  = pam[0:2]  if len(pam) >= 2 else ""
    junc   = (full23[19] + pam[0]) if len(full23) >= 21 else ""
    pam12  = pam[1:3]  if len(pam) >= 3 else ""
    for d in DINUCS:
        f[f"t6_di_pam01_{d}"]    = int(pam01 == d)
        f[f"t6_di_junc2021_{d}"] = int(junc == d)
        f[f"t6_di_pam12_{d}"]    = int(pam12 == d)

    # Trinucs at pos 18-20 (spacer[17:20]) and pos 20-22 (full23[19:22])
    tri_1820 = full23[17:20] if len(full23) >= 20 else ""
    tri_2022 = full23[19:22] if len(full23) >= 22 else ""
    for t in TRINUCS:
        f[f"t6_tri_1820_{t}"] = int(tri_1820 == t)
        f[f"t6_tri_2022_{t}"] = int(tri_2022 == t)

    # Seed-PAM G-run binary (GG at 3 positions, GGG at 3 positions)
    f["t6_GG_1819"]  = int(len(full23) >= 19 and full23[17:19] == "GG")
    f["t6_GG_1920"]  = int(len(full23) >= 20 and full23[18:20] == "GG")
    f["t6_GG_2021"]  = int(len(full23) >= 21 and full23[19:21] == "GG")
    f["t6_GGG_1820"] = int(len(full23) >= 20 and full23[17:20] == "GGG")
    f["t6_GGG_1921"] = int(len(full23) >= 21 and full23[18:21] == "GGG")
    f["t6_GGG_2022"] = int(len(full23) >= 22 and full23[19:22] == "GGG")

    # TT penalty at pos 19-20 (spacer[18:20])
    f["t6_TT_1920"] = int(len(full23) >= 20 and full23[18:20] == "TT")
    return f


# GC, melting temperature, and local stability

def _module9_row(spacer, full23):
    f = {}
    seed_seq  = spacer[12:20]
    pampx_seq = full23[16:23]
    fivep_seq = spacer[0:10]
    mid_seq   = spacer[6:16]

    f["t7_tm_full"]  = wallace_tm(spacer)
    f["t7_tm_seed"]  = wallace_tm(seed_seq)
    f["t7_tm_pampx"] = wallace_tm(pampx_seq)
    f["t7_tm_5p"]    = wallace_tm(fivep_seq)
    f["t7_tm_mid"]   = wallace_tm(mid_seq)

    f["t7_dg_full"]  = dg_proxy(spacer)
    f["t7_dg_seed"]  = dg_proxy(seed_seq)
    f["t7_dg_pampx"] = dg_proxy(pampx_seq)

    # Sliding window GC (k=4, 17 windows on 20-nt spacer)
    sw_gcs = []
    for i in range(17):
        w = spacer[i:i+4]
        gc_w = (w.count("G") + w.count("C")) / 4.0
        f[f"t7_swgc_w{i}"] = gc_w
        sw_gcs.append(gc_w)
        f[f"t7_swtm_w{i}"] = wallace_tm(w)

    f["t7_dtm_seed_5p"] = f["t7_tm_seed"] - f["t7_tm_5p"]
    f["t7_swgc_min"]    = min(sw_gcs)
    f["t7_swgc_max"]    = max(sw_gcs)
    return f


# Secondary-structure proxies

def _module10_row(spacer):
    f = {}
    f["t8_selfcomp_count"] = self_comp_count(spacer)
    f["t8_selfcomp_max4"]  = max_self_comp_window(spacer, 4)
    f["t8_selfcomp_max6"]  = max_self_comp_window(spacer, 6)
    f["t8_selfcomp_max8"]  = max_self_comp_window(spacer, 8)
    half = len(spacer) // 2
    gc5 = spacer[:half].count("G") + spacer[:half].count("C")
    gc3 = spacer[half:].count("G") + spacer[half:].count("C")
    f["t8_gc_asym"] = gc5 - gc3
    return f


# Homopolymer and low-complexity features

def _module11_row(spacer):
    f = {}
    for nt in NTS:
        f[f"t9_max_{nt}_run"] = max_run_of(spacer, nt)
    f["t9_has_T4"]      = int(max_run_of(spacer, "T") >= 4)
    f["t9_has_G4"]      = int(max_run_of(spacer, "G") >= 4)
    f["t9_T_run_count"] = count_runs_of(spacer, "T", 2)
    f["t9_G_run_count"] = count_runs_of(spacer, "G", 2)
    for pat in ["TATA", "ACAC", "CGCG", "ATAT"]:
        f[f"t9_direpeat_{pat}"] = count_nonoverlap(spacer, pat)
    f["t9_entropy"]    = shannon_entropy(spacer)
    f["t9_low_complex"]= int(shannon_entropy(spacer) < 1.5)
    f["t9_5p_A_count"] = spacer[0:5].count("A")
    return f


# Interaction features

def _module12_row(spacer, full23, fm7, fm9, fm11):
    f = {}
    pam = full23[20:23]
    pam_TGG = int(pam == "TGG")
    pam_CGG = int(pam == "CGG")
    pam_AGG = int(pam == "AGG")
    pam_GGG = int(pam == "GGG")
    gc_seed = fm7["t5_seed_gc_frac"]
    gc_full = fm7["t5_fullsp_gc_frac"]
    pos20_G = int(len(spacer) >= 20 and spacer[19] == "G")

    f["t10_pam_TGG_x_gc_seed"] = pam_TGG * gc_seed
    f["t10_pam_CGG_x_gc_seed"] = pam_CGG * gc_seed
    f["t10_pam_AGG_x_gc_seed"] = pam_AGG * gc_seed
    f["t10_pam_GGG_x_gc_seed"] = pam_GGG * gc_seed

    f["t10_pam_TGG_x_pos20G"] = pam_TGG * pos20_G
    f["t10_pam_CGG_x_pos20G"] = pam_CGG * pos20_G
    f["t10_pam_AGG_x_pos20G"] = pam_AGG * pos20_G
    f["t10_pam_GGG_x_pos20G"] = pam_GGG * pos20_G

    # Seed GC by gapped TG/GG counts
    ngd1 = ngd_counts_in(full23, 1)
    f["t10_gc_seed_x_ngd_TG1"] = gc_seed * ngd1.get("TG", 0)
    f["t10_gc_seed_x_ngd_GG1"] = gc_seed * ngd1.get("GG", 0)

    f["t10_max_G_run_x_pam_TGG"]  = fm11["t9_max_G_run"] * pam_TGG
    f["t10_max_T_run_x_gc_seed"]  = fm11["t9_max_T_run"] * gc_seed
    f["t10_entropy_x_gc_frac"]    = fm11["t9_entropy"] * gc_full
    f["t10_tm_seed_x_tm_5p"]      = fm9["t7_tm_seed"] * fm9["t7_tm_5p"]

    pos19_G = int(len(spacer) >= 19 and spacer[18] == "G")
    pos21_G = int(len(full23) >= 21 and full23[20] == "G")
    pos19_T = int(len(spacer) >= 19 and spacer[18] == "T")
    pos20_T = int(len(spacer) >= 20 and spacer[19] == "T")
    f["t10_GG_junc"]    = pos20_G * pos21_G
    f["t10_GGG_junc"]   = pos19_G * pos20_G * pos21_G
    f["t10_TT_penalty"] = pos19_T * pos20_T
    return f


def _frame(records):
    return pd.DataFrame(records, dtype=float)


def extract_module4(spacers, full23):
    return _frame([_module4_row(spacer) for spacer in spacers])


def extract_module5(spacers, full23):
    return _frame([_module5_row(spacer) for spacer in spacers])


def extract_module6(spacers, full23):
    return _frame([_module6_row(spacer, seq) for spacer, seq in zip(spacers, full23)])


def extract_module7(spacers, full23):
    return _frame([_module7_row(spacer, seq) for spacer, seq in zip(spacers, full23)])


def extract_module8(spacers, full23):
    return _frame([_module8_row(spacer, seq) for spacer, seq in zip(spacers, full23)])


def extract_module9(spacers, full23):
    return _frame([_module9_row(spacer, seq) for spacer, seq in zip(spacers, full23)])


def extract_module10(spacers, full23):
    return _frame([_module10_row(spacer) for spacer in spacers])


def extract_module11(spacers, full23):
    return _frame([_module11_row(spacer) for spacer in spacers])


def extract_module12(spacers, full23):
    records = []
    for spacer, seq in zip(spacers, full23):
        records.append(
            _module12_row(
                spacer,
                seq,
                _module7_row(spacer, seq),
                _module9_row(spacer, seq),
                _module11_row(spacer),
            )
        )
    return _frame(records)


MODULE_EXTRACTORS = {
    f"module{number}": globals()[f"extract_module{number}"]
    for number in range(1, 13)
}


def extract_module(module, spacers, full23):
    try:
        extractor = MODULE_EXTRACTORS[module]
    except KeyError as exc:
        raise ValueError(f"Unknown feature module: {module!r}") from exc
    return extractor(spacers, full23)
