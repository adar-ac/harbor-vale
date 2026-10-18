# Model card: Telco churn classifier (dry-run)

Generated without an LLM; numbers come from `crews/scientist/tools.py`.

## Purpose
- Estimate the probability that a Telco customer churns, to prioritise retention outreach.

## Model
- Selected: `random_forest` (RandomForestClassifier), file `model.pkl`
- Selection: cv_roc_auc_mean: `random_forest` 0.8468 vs `logistic_regression` 0.846 (margin 0.0008); a near tie (under 0.01), so the choice could reasonably be made on other grounds such as interpretability
- Most influential features: `tenure`, `TotalCharges`, `Contract_Two year`, `MonthlyCharges`, `InternetService_Fiber optic`

## Training data
- Source: clean.csv, dataset_contract.json (contract v1.0, produced by crew1_analyst)
- Rows: 7032; class counts: {'No': 5163, 'Yes': 1869}; churn rate 26.58%
- Dropped identifier columns: ['customerID']
- Features after encoding: 30

## Metrics (hold-out)

| Model | precision | recall | f1 | roc_auc | accuracy | cv_roc_auc_mean | selected |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.4909 | 0.7968 | 0.6075 | 0.8350 | 0.7264 | 0.8460 |  |
| random_forest | 0.5195 | 0.7834 | 0.6247 | 0.8367 | 0.7498 | 0.8468 | yes |

Confusion matrix, logistic_regression (hold-out, threshold 0.5): TN=724 FP=309 FN=76 TP=298

Confusion matrix, random_forest (hold-out, threshold 0.5): TN=762 FP=271 FN=81 TP=293

## Limitations
- Single static snapshot; no time-based validation, so drift is untested.
- Threshold fixed at 0.5 with balanced class weights; precision/recall trade-off not tuned.
- Customers with tenure 0 were removed in cleaning, so new customers are out of scope.

## Ethical considerations
- Inputs include gender and SeniorCitizen; check outcomes per group before acting on scores.
- Use scores to offer help, not to penalise customers or withhold service.
