# Model evaluation: logistic regression vs random forest (dry-run)

Generated without an LLM; numbers come from `crews/scientist/tools.py` (metrics.json).

## Setup
- Split: stratified hold-out, test_size=0.2, random_state=42
- Train rows: 5625, test rows: 1407, features: 30
- Churn rate: train 0.2658, test 0.2658
- Selection: highest cv_roc_auc_mean (5-fold CV on the training split)
- Decision threshold: 0.5; positive class: Yes (churned) = 1

## Hold-out metrics

| Model | precision | recall | f1 | roc_auc | accuracy | cv_roc_auc_mean | selected |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.4909 | 0.7968 | 0.6075 | 0.8350 | 0.7264 | 0.8460 |  |
| random_forest | 0.5195 | 0.7834 | 0.6247 | 0.8367 | 0.7498 | 0.8468 | yes |

Confusion matrix, logistic_regression (hold-out, threshold 0.5): TN=724 FP=309 FN=76 TP=298

Confusion matrix, random_forest (hold-out, threshold 0.5): TN=762 FP=271 FN=81 TP=293

## Selected model
- `random_forest`, saved as `model.pkl`
- cv_roc_auc_mean: `random_forest` 0.8468 vs `logistic_regression` 0.846 (margin 0.0008); a near tie (under 0.01), so the choice could reasonably be made on other grounds such as interpretability
