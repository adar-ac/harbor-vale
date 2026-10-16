# Evaluation Report

## Setup

The models were trained using a stratified hold-out method with a test size of 20%. The random state was set to 42 to ensure reproducibility. Cross-validation was performed with 5 folds, and the selection metric for model comparison was the mean ROC-AUC score. The decision threshold for classification was set at 0.5, and the positive class was defined as "Yes (churned) = 1".

## Metrics

| Model | precision | recall | f1 | roc_auc | accuracy | cv_roc_auc_mean | selected |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.4909 | 0.7968 | 0.6075 | 0.8350 | 0.7264 | 0.8460 |  |
| random_forest | 0.5195 | 0.7834 | 0.6247 | 0.8367 | 0.7498 | 0.8468 | yes |

Confusion matrix, logistic_regression (hold-out, threshold 0.5): TN=724 FP=309 FN=76 TP=298

Confusion matrix, random_forest (hold-out, threshold 0.5): TN=762 FP=271 FN=81 TP=293

## Discussion of Precision vs Recall

In the context of churn prediction, precision and recall are critical metrics. Precision indicates the proportion of true positive predictions among all positive predictions, while recall measures the proportion of true positives identified among all actual positives. 

- **Logistic Regression** achieved a precision of 0.4909 and a recall of 0.7968, indicating that while it correctly identifies a significant number of churned customers, it also misclassifies many non-churned customers as churned.
- **Random Forest** performed slightly better with a precision of 0.5195 and a recall of 0.7834. This suggests that it is more accurate in identifying churned customers while still maintaining a reasonable recall.

The confusion matrices for both models reveal the trade-offs between false positives and false negatives. For Logistic Regression, there were 724 true negatives, 309 false positives, 76 false negatives, and 298 true positives. This indicates a higher number of false positives, which can lead to unnecessary outreach efforts. In contrast, the Random Forest model had 762 true negatives, 271 false positives, 81 false negatives, and 293 true positives, demonstrating a more balanced performance with fewer false positives.

## Why Accuracy Alone is Misleading

Given the churn rate of approximately 26.58% in both training and testing datasets, accuracy alone can be misleading. A model could achieve high accuracy by simply predicting the majority class (non-churned customers) without effectively identifying the minority class (churned customers). Therefore, relying solely on accuracy would not provide a true reflection of the model's performance in predicting churn.

## Model Comparison

The gap between the models is relatively small, with Random Forest slightly outperforming Logistic Regression in terms of both F1 score (0.6247 vs. 0.6075) and ROC-AUC (0.8367 vs. 0.835). However, the Random Forest model was selected due to its marginally higher ROC-AUC score and better overall performance in precision and recall.

## Important Features

### Logistic Regression
1. **Tenure**: -1.2639
2. **MonthlyCharges**: -0.9287
3. **InternetService_Fiber optic**: 0.7461
4. **Contract_Two year**: -0.6386
5. **TotalCharges**: 0.6233

### Random Forest
1. **Tenure**: 0.1837
2. **TotalCharges**: 0.1386
3. **Contract_Two year**: 0.1056
4. **MonthlyCharges**: 0.1
5. **InternetService_Fiber optic**: 0.0683

The most important features for both models include "tenure" and "TotalCharges," indicating that these factors are significant predictors of customer churn.

## Selected Model

The selected model is the **Random Forest** due to its slightly superior performance in ROC-AUC and its ability to balance precision and recall more effectively than Logistic Regression. This makes it a more reliable choice for identifying churned customers and optimizing outreach efforts.
