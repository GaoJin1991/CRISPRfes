# CRISPRfes

CRISPR Feature Engineering and Selection for sequence-based sgRNA activity prediction.

## Contents

- `CRISPRfes/`: all 12 feature modules, feature selection, and SVM models
- `models/`: 23 pretrained models with JSON metadata
- `data/`: human classification, human regression, zebrafish, and supplementary HEK293/HEK293T data
- `results/`: manuscript results and external-tool scores
- `tables/`: Supplementary Tables S1–S5 in DOCX, XLSX, and PDF
- `figures/`: five manuscript figures in SVG, PDF, PNG, and TIFF
- `tests/`: data, feature, model, and command-line checks

Figure 1 can be regenerated from its [editable source and build script](figures/source/README.md).

## Setup

Use Python 3.8–3.11 and the pinned dependencies. Python 3.12+ is not supported by these dependencies. The original environment uses Python 3.8.20 and scikit-learn 1.3.2. Open a terminal in the project root, which contains `requirements.txt`, before running commands.

Choose one setup method:

### Standard Python

On Windows, install a standard Python 3.11 distribution and use PowerShell:

```powershell
py -3.11 -m venv CRISPRfes_env
.\CRISPRfes_env\Scripts\python.exe -m pip install -r requirements.txt
.\CRISPRfes_env\Scripts\python.exe -m pip check
.\CRISPRfes_env\Scripts\python.exe -m unittest discover -s tests -v
```

If `py` is unavailable, use the full path to a standard Python 3.8–3.11 executable. Do not use a Conda interpreter to create this virtualenv; use the Conda method below instead.

On Linux/macOS:

```bash
python3.11 -m venv CRISPRfes_env
CRISPRfes_env/bin/python -m pip install -r requirements.txt
CRISPRfes_env/bin/python -m pip check
CRISPRfes_env/bin/python -m unittest discover -s tests -v
```

In subsequent commands, replace `python` with `.\CRISPRfes_env\Scripts\python.exe` on Windows or `CRISPRfes_env/bin/python` on Linux/macOS. Activation is not required for this method.

### Conda

Use an Anaconda/Miniconda prompt or a shell configured for Conda:

```bash
conda create -n CRISPRfes_env python=3.11 pip -y
conda activate CRISPRfes_env
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
```

Keep this environment activated when running the commands below.

### PyCharm

Open the project root and configure the project interpreter before installing dependencies. Select a standard Python 3.8–3.11 as the base interpreter and name the new virtualenv `CRISPRfes_env`; do not accept an incompatible default interpreter. Alternatively, create the Conda environment above and select it as an existing Conda interpreter. Install `requirements.txt` using the chosen environment.

## Predict

Run from the project directory. Training is not required.

```bash
python predict.py --model models/CRISPRfes_zebrafish_G3.joblib --input examples/example_input.csv --out predictions/zebrafish.csv
```

For human data, use `CRISPRfes_human_classification_G3.joblib` or `CRISPRfes_human_regression_G3.joblib`. Models containing `_holdout_` exclude the named cell line. See [models/README.md](models/README.md).

Input is CSV or XLSX, with one sequence per row:

```csv
sgRNA
CTTGCTCGCGCAGGACGAGGCGG
ACATCAGGTTACCTCTACCAAGG
```

Supply a 23-nt DNA target (20-nt spacer plus its 3-nt PAM), or a 20-nt spacer with a separate `PAM` column. Use `--seq-col` and `--pam-col` for other column names. Lowercase bases are accepted. Missing PAMs are not inferred; an explicit `N` in a PAM is accepted as unknown and is not equivalent to a known base.

Output retains input columns and adds `CRISPRfes_score`. Larger values indicate higher predicted activity. Classification returns SVC decision values, not probabilities. Regression uses the training-label scale; zebrafish models return standardized values by default. Add `--original-scale` to undo stored label z-scoring. This does not calibrate scores across assays or constrain them to a percentage range. `--batch-size` defaults to 256.

## Train and evaluate

Train a classification model excluding HCT116:

```bash
python train.py --data data/human_classification.csv --task classification --group G3 --group-col cell --hold-out HCT116 --out models/custom_hct116.joblib
```

Evaluate human classification or regression:

```bash
python evaluate.py --data data/human_classification.csv --task classification --protocol loco --group-col cell --group G3 --out evaluation/classification.csv
python evaluate.py --data data/human_regression.csv --task regression --protocol loco --group-col cell --group G3 --out evaluation/regression.csv
```

Each held-out cell line has five rounds. Round k tests its kth fold and trains on the other cell lines with their kth folds excluded. Stability is summarized separately for each held-out cell line.

Run all three groups on zebrafish data:

```bash
python reproduce_three_groups.py --data data/zebrafish_crisprz.csv --task regression --protocol kfold --zscore --outdir evaluation/zebrafish
```

G1 enables FM1–FM3 with embedded selection, G2 enables FM1–FM12 with embedded selection, and G3 uses FM1–FM12 with stable-core selection. Settings are in `CRISPRfes/config.py`; all modules are implemented in `CRISPRfes/feature_modules.py`. G3 defaults to τ = 0.90 and B = 40 for classification or 50 for regression. No features meeting τ produces an error rather than an unreported fallback. Summary standard deviations use `ddof=1`.

## Data

| File | Samples | Response |
|---|---:|---|
| `human_classification.csv` | 14,416 | binary activity label |
| `human_regression.csv` | 14,416 | normalized efficacy |
| `hek293_classification.csv` | 2,333 | binary activity label |
| `hek293t_regression.csv` | 2,333 | normalized efficacy |
| `zebrafish_crisprz.csv` | 1,173 | somatic activity |

Human classification and regression are separate analysis datasets. Human data derive from resources used by Haeussler et al., DeepCRISPR, and CRISPRpred(SEQ). Zebrafish data were curated from the [NIH CRISPRz database](https://research.nhgri.nih.gov/crisprz/). Labels are stored as supplied; `--zscore` normalizes training labels using training statistics only. Zebrafish evaluation uses shuffled five-fold cross-validation without grouping by gene. Follow the original sources' terms when reusing data.

Full-data models are for new inputs, not independent test estimates on their training rows. Scripts implement the main G1/G2/G3 comparisons; dimension-matched controls, threshold sweeps, and external-tool scores remain in the supplied results rather than a single automated reproduction command.

## References

- Chuai G. et al. DeepCRISPR. *Genome Biology*. 2018;19:80.
- Rafid A.H. et al. CRISPRpred(SEQ). *BMC Bioinformatics*. 2020;21:223.
- Varshney G.K. et al. CRISPRz: a database of zebrafish validated sgRNAs. *Nucleic Acids Research*. 2016;44(D1):D822–D826. https://doi.org/10.1093/nar/gkv998.
