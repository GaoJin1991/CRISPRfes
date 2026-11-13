# Pretrained models

Each `.joblib` contains a fitted SVM, scaler, feature names, and selected-column indices. Its paired JSON provides training metadata; it is not a prediction model by itself.

| Use | Model |
|---|---|
| Human classification | `CRISPRfes_human_classification_G3.joblib` |
| Human regression | `CRISPRfes_human_regression_G3.joblib` |
| Zebrafish regression | `CRISPRfes_zebrafish_G3.joblib` |

The two general human G3 models use all 14,416 rows in their respective tasks. Three zebrafish G1/G2/G3 models use all 1,173 rows. Eighteen human G1/G2/G3 models named `_holdout_HCT116/HeLa/HL60` exclude the named cell line. These are full training-pool models, not the individual five-fold models used for manuscript estimates.

From the project root:

```bash
python predict.py --model models/CRISPRfes_human_classification_G3.joblib --input examples/example_input.csv --out predictions/human_classification.csv
python predict.py --model models/CRISPRfes_human_regression_G3.joblib --input examples/example_input.csv --out predictions/human_regression.csv
python predict.py --model models/CRISPRfes_zebrafish_G3.joblib --input examples/example_input.csv --out predictions/zebrafish.csv
```

`CRISPRfes_score` is an SVC decision value for classification and a predicted response for regression. Zebrafish models use z-scored labels; `--original-scale` reverses stored normalization. Scores are not calibrated probabilities or guaranteed editing percentages.

Use `requirements.txt` to match the serialized models. Only load trusted `.joblib` files. Historical `source_data` names identify original training inputs and are not runtime paths. Current bundles use `CRISPRfes-model-v1`. The loader also accepts earlier model formats.
