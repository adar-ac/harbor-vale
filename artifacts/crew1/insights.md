# Insights on Customer Churn

## Strongest Churn Drivers
The following segments exhibit the highest churn rates compared to the overall churn rate of **26.58%**:

- **Senior Citizens**: 
  - Churn Rate: **41.68%** (compared to **23.65%** for non-seniors)
  
- **Customers without a Partner**: 
  - Churn Rate: **32.98%** (compared to **19.72%** for those with a partner)
  
- **Customers without Dependents**: 
  - Churn Rate: **31.28%** (compared to **15.53%** for those with dependents)
  
- **Contract Type**:
  - Month-to-month: **42.71%**
  - One year: **11.28%**
  - Two year: **2.85%**
  
- **Payment Method**:
  - Electronic check: **45.29%**
  - Bank transfer (automatic): **16.73%**
  - Credit card (automatic): **15.25%**
  - Mailed check: **19.20%**

## Notable Numeric Differences
### Churned vs. Retained Customers
- **Tenure**:
  - Mean for churned: **17.98 months**
  - Mean for retained: **37.65 months**
  
- **Monthly Charges**:
  - Mean for churned: **74.44 USD**
  - Mean for retained: **61.31 USD**
  
- **Total Charges**:
  - Mean for churned: **1531.8 USD**
  - Mean for retained: **2555.34 USD**

## Data Quality Notes
- The dataset originally contained **7043 rows**, with **11 rows dropped** due to blank values in the `TotalCharges` column, resulting in a final count of **7032 rows**.
- The `TotalCharges` column was converted from text to numeric (`float64`), with a minimum value of **18.8 USD** and a maximum of **8684.8 USD**.

## Caveats
- **Class Balance**: The dataset has a churn rate of **26.58%**, indicating a moderate imbalance that may affect model performance.
- **Correlated Features**: 
  - There is a potential correlation between `tenure` and `TotalCharges`, which may need to be addressed during modeling to avoid multicollinearity issues.
