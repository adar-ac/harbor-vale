# Evaluation Report

## Setup
The models were trained using a stratified hold-out split with a test size of 20%. The random state was set to 42 to ensure reproducibility. Cross-validation was performed using 5 folds. The decision threshold for classification was set at 0.5, and the positive class was defined as "Yes (churned) = 1". Class weighting was applied to both models to address the imbalance in the churn data.

## Metrics Table
| Model | precision | recall | f1 | roc_auc | accuracy | cv_roc_auc_mean | selected |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.4909 | 0.7968 | 0.6075 | 0.8350 | 0.7264 | 0.8460 |  |
| random_forest | 0.5195 | 0.7834 | 0.6247 | 0.8367 | 0.7498 | 0.8468 | yes |

Confusion matrix, logistic_regression (hold-out, threshold 0.5): TN=724 FP=309 FN=76 TP=298

Confusion matrix, random_forest (hold-out, threshold 0.5): TN=762 FP=271 FN=81 TP=293

## Discussion of Precision vs Recall
In the context of churn prediction, precision and recall are critical metrics. Precision indicates the proportion of true positive predictions among all positive predictions, while recall measures the proportion of true positives identified out of all actual positives. 

- **Logistic Regression** achieved a precision of 0.4909 and a recall of 0.7968. This indicates that while it captures a good number of actual churners, nearly half of its positive predictions are incorrect.
- **Random Forest** performed slightly better with a precision of 0.5195 and a recall of 0.7834. This suggests it is somewhat more reliable in identifying churners, but still has a significant number of false positives.

The confusion matrices for both models reveal the challenges in accurately predicting churn. For Logistic Regression, there were 298 true positives, 724 true negatives, 76 false negatives, and 309 false positives. For Random Forest, the confusion matrix showed 293 true positives, 762 true negatives, 81 false negatives, and 271 false positives. The higher number of false positives in both models indicates a risk of unnecessary outreach to customers who are not likely to churn.

## Why Accuracy Alone is Misleading
Accuracy can be misleading in imbalanced datasets like churn prediction, where the majority of customers do not churn. For instance, if 73.42% of customers do not churn, a model that predicts all customers as non-churners could achieve a high accuracy of 73.42%, despite failing to identify any actual churners. Therefore, relying solely on accuracy does not provide a complete picture of model performance in this context.

## Model Comparison
The gap between the models is minimal, with Random Forest achieving a mean CV ROC-AUC of 0.8468 compared to Logistic Regression's 0.846. The margin of 0.0008 indicates a near tie, suggesting that both models are effectively equal on the selection metric. 

## Important Features
The most important features identified by each model are as follows:

- **Logistic Regression**:
  - Tenure (standardised coefficient: -1.2639)
  - MonthlyCharges (standardised coefficient: -0.9287)
  - InternetService_Fiber optic (standardised coefficient: 0.7461)

- **Random Forest**:
  - Tenure (importance: 0.1837)
  - TotalCharges (importance: 0.1386)
  - Contract_Two year (importance: 0.1056)

## Selected Model
The selected model is the **Random Forest**, based on its higher mean CV ROC-AUC of 0.8468 compared to Logistic Regression's 0.846, with a margin of 0.0008. Given that the two models are effectively equal on the selection metric, the choice could reasonably be made on other grounds, such as interpretability. Logistic Regression coefficients are easier to explain than the complex structure of a Random Forest, which may be a consideration for stakeholders seeking clarity in model decisions.
