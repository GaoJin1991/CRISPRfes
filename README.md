# SgFES

Code, datasets, result tables, and figures for the SgFES sgRNA activity study.

## Contents

- `sgfes/`: FM1-FM12 extraction, selection, and SVM models
- `data/`: five non-redundant analysis tables
- `results/`: SgFES and external-tool values reported in the manuscript
- `tables/`: two main tables and Supplementary Tables S1-S3
- `figures/`: manuscript figures in SVG, PDF, and PNG
- `models/`: pretrained model bundles and metadata
- `tests/`: integrity and pipeline tests

## Setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Python 3.9-3.11 is recommended. The pretrained models were serialized with scikit-learn 1.3.2.

All feature modules are implemented in `sgfes/feature_modules.py`. The `GROUPS` settings in `sgfes/config.py` select the active modules for each experiment; G1 enables FM1-FM3, while G2 and G3 enable FM1-FM12.

## Use

Train a model:

```bash
python train.py --data data/human_classification.csv --task classification \
  --group G3 --group-col cell --hold-out HCT116 --out models/hct116.joblib
```

Score sequences:

```bash
python predict.py --model models/hct116.joblib \
  --input examples/example_input.csv --out predictions.csv
```

Run the human cross-cell protocol:

```bash
python evaluate.py --data data/human_classification.csv \
  --task classification --protocol loco --group-col cell --group G3
```

For each held-out cell line, the cross-cell protocol tests one of its five folds and trains on the corresponding training folds from the other cell lines. This gives 15 evaluations across HCT116, HeLa, and HL60. The zebrafish analysis uses shuffled five-fold cross-validation:

```bash
python evaluate.py --data data/zebrafish_crisprz.csv \
  --task regression --protocol kfold --zscore --group G3
```

## Data

| File | Samples | Response |
|---|---:|---|
| `human_classification.csv` | 14,416 | binary activity label |
| `human_regression.csv` | 14,416 | normalized efficacy |
| `hek293_classification.csv` | 2,333 | binary activity label |
| `hek293t_regression.csv` | 2,333 | normalized efficacy |
| `zebrafish_crisprz.csv` | 1,173 | somatic activity |

All guide sequences are 23 nt (20-nt spacer plus 3-nt PAM). The human tables derive from the datasets used by Haeussler et al., DeepCRISPR, and CRISPRpred(SEQ). The zebrafish table was cleaned from the [NIH CRISPRz database](https://research.nhgri.nih.gov/crisprz/). Labels were retained as published. Follow the original sources' terms when reusing the data.

Human classification labels and human regression responses are stored separately to avoid duplicating per-cell-line files. HEK293/HEK293T is retained as a separate supplementary dataset. The zebrafish protocol is ordinary five-fold cross-validation and does not group guides by gene.

Each pretrained model in `models/` is paired with a JSON metadata file. See `models/README.md` for the general-use and evaluation models.

## References

- Chuai G. et al. DeepCRISPR. *Genome Biology*. 2018;19:80.
- Rafid A.H. et al. CRISPRpred(SEQ). *BMC Bioinformatics*. 2020;21:223.
- Varshney G.K. et al. CRISPRz. *Genome Biology*. 2016;17:83.
