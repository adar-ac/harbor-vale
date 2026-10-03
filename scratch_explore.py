import pandas as pd
df = pd.read_csv("data/raw/telco.csv")
print(df.shape)
print(df.dtypes)
print(df["TotalCharges"].describe())
print(df["Churn"].value_counts())
print("--------------------------------")
blanks = df[pd.to_numeric(df["TotalCharges"], errors="coerce").isna()]
print(len(blanks))
print(blanks[["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])