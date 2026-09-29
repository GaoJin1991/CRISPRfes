"""Model and feature settings used in the SgFES experiments."""

SVM_C = 10.0
SVM_GAMMA = 0.001

N_EST = 200
SEED = 42
TAU_DEFAULT = 0.90
TAU_GRID = (0.80, 0.85, 0.90, 0.95)
B_CLASSIFICATION = 40
B_REGRESSION = 50
SUBSAMPLE_RATIO = 0.5

MODULES = [
    ("module1", "position-independent k-mers", 340),
    ("module2", "position-specific encoding", 6908),
    ("module3", "gapped dinucleotides", 336),
    ("module4", "spacer k-mers", 370),
    ("module5", "extended positional encoding", 426),
    ("module6", "regional gapped dinucleotides", 480),
    ("module7", "regional summary statistics", 135),
    ("module8", "PAM-proximal features", 192),
    ("module9", "thermodynamic proxies", 45),
    ("module10", "secondary-structure proxies", 5),
    ("module11", "homopolymers and complexity", 15),
    ("module12", "interaction terms", 17),
]
MODULE_IDS = [m[0] for m in MODULES]
TOTAL_DIM = sum(m[2] for m in MODULES)


def module_label(mid):
    name = {m[0]: m[1] for m in MODULES}[mid]
    return f"FM{mid.replace('module', '')} ({name})"


GROUPS = {
    "G1": {
        "modules": MODULE_IDS[:3],
        "selection": "broad",
        "desc": "FM1-FM3 with embedded selection",
    },
    "G2": {
        "modules": MODULE_IDS[:],
        "selection": "broad",
        "desc": "FM1-FM12 with embedded selection",
    },
    "G3": {
        "modules": MODULE_IDS[:],
        "selection": "stability",
        "desc": "FM1-FM12 with stability selection",
    },
}
DEFAULT_GROUP = "G3"


def default_B(task):
    return B_CLASSIFICATION if task == "classification" else B_REGRESSION
