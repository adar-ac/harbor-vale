import pandas as pd, json

df = pd.read_csv("data/raw/telco.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


df = df.dropna(subset=["TotalCharges"])
cols_to_describe = [c for c in df.columns if c != "customerID"]


cols = {}
for c in cols_to_describe:
    if pd.api.types.is_numeric_dtype(df[c]):
        cols[c] = {"dtype": str(df[c].dtype),
                   "min": float(df[c].min()), "max": float(df[c].max())}
    else:
        cols[c] = {"dtype": "str",
                   "allowed": sorted(df[c].unique().tolist())}

print(len(df))
with open("scratch_facts.json", "w") as f:
    json.dump(cols, f, indent=2)