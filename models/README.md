# Pretrained models

Each `.joblib` model is paired with a JSON metadata file containing its task, feature group, selected columns, and training settings.

General human-cell models:

- `sgfes_human_classification_G3.joblib`
- `sgfes_human_regression_G3.joblib`

Files containing `_holdout_` are the cell-line holdout models used for evaluation. Zebrafish models are provided for feature groups G1, G2, and G3.

Run a model on new sequences with:

```bash
python predict.py --model models/sgfes_human_classification_G3.joblib --input examples/example_input.csv --out predictions.csv
```

Install the dependencies in `requirements.txt` before prediction.
