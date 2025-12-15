import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("topsis_fintech_panel_2010_2024.csv")
df = df[df["year"] <= 2023].reset_index(drop=True)

ind_cols = [c for c in df.columns if c not in ["country_code", "country_name", "year"]]

df = df.sort_values(["country_code", "year"])

for col in ind_cols:
    df[f"{col}_growth"] = df.groupby("country_code")[col].pct_change(fill_method=None)

for col in ind_cols:
    df[f"{col}_stab"] = df.groupby("country_code")[col].transform(
        lambda x: 1.0 / (x.rolling(window=3, min_periods=2).std() + 1e-6)
    )
num_cols = df.select_dtypes(include=[np.number]).columns

df[num_cols] = (
    df.groupby("country_code")[num_cols]
      .apply(lambda x: x.interpolate())
      .reset_index(level=0, drop=True)
)

df[num_cols] = df[num_cols].ffill().bfill()
all_cols = (
    ind_cols
    + [c + "_growth" for c in ind_cols]
    + [c + "_stab" for c in ind_cols]
)

X = df[all_cols].values.astype(float)
m, n = X.shape

min_vals = X.min(axis=0)
max_vals = X.max(axis=0)
range_vals = max_vals - min_vals
range_vals[range_vals == 0] = 1.0

R = (X - min_vals) / range_vals

W = np.ones(n) / n

V = R * W

A_plus = V.max(axis=0)
A_minus = V.min(axis=0)

S_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
S_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))

C = S_minus / (S_plus + S_minus + 1e-12)

df_result = df[["country_code", "country_name", "year"]].copy()
df_result["topsis_3d"] = C

print(df_result)

plt.figure(figsize=(12, 6))
for code in sorted(df_result["country_code"].unique()):
    sub = df_result[df_result["country_code"] == code]
    plt.plot(sub["year"], sub["topsis_3d"], marker="o", label=code)

plt.xlabel("Year")
plt.ylabel("TOPSIS 3D score (Level + Growth + Stability)")
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()
