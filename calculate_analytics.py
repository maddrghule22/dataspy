import pandas as pd
import numpy as np
import json

df = pd.read_csv("dataset/train.csv")

# Overall stats
overall_stats = {
    "total_records": int(len(df)),
    "min_price": float(df["SalePrice"].min()),
    "max_price": float(df["SalePrice"].max()),
    "median_price": float(df["SalePrice"].median()),
    "mean_price": float(df["SalePrice"].mean()),
    "median_gr_liv_area": float(df["GrLivArea"].median()),
    "median_overall_qual": float(df["OverallQual"].median()),
    "median_total_bsmt_sf": float(df["TotalBsmtSF"].median()),
    "median_garage_cars": float(df["GarageCars"].median()),
    "median_year_built": float(df["YearBuilt"].median()),
}

# Neighborhood stats
neigh_grouped = df.groupby("Neighborhood")["SalePrice"].agg(
    count="count",
    mean="mean",
    median="median",
    min="min",
    max="max"
).reset_index()

neigh_dict = {}
for _, row in neigh_grouped.iterrows():
    neigh_dict[row["Neighborhood"]] = {
        "count": int(row["count"]),
        "mean": float(row["mean"]),
        "median": float(row["median"]),
        "min": float(row["min"]),
        "max": float(row["max"])
    }

# Quality vs Price
qual_grouped = df.groupby("OverallQual")["SalePrice"].agg(
    count="count",
    mean="mean",
    median="median"
).reset_index().to_dict(orient="records")

# Living area bins vs Price
df['AreaBin'] = pd.cut(df['GrLivArea'], bins=[0, 1000, 1500, 2000, 2500, 5000], labels=['<1k', '1k-1.5k', '1.5k-2k', '2k-2.5k', '>2.5k'])
area_grouped = df.groupby("AreaBin", observed=False)["SalePrice"].agg(
    count="count",
    median="median"
).reset_index()
area_dict = {str(k): {"count": int(v["count"]), "median": float(v["median"])} for k, v in area_grouped.set_index("AreaBin").to_dict(orient="index").items()}

# Sample raw records for dataset explorer (first 10 records)
explorer_records = df[['Id', 'Neighborhood', 'SalePrice', 'GrLivArea', 'OverallQual', 'YearBuilt', 'GarageCars', 'TotalBsmtSF']].head(15).to_dict(orient="records")

analytics_data = {
    "overall": overall_stats,
    "neighborhoods": neigh_dict,
    "quality": qual_grouped,
    "area": area_dict,
    "explorer_records": explorer_records
}

with open("dataset_analytics.json", "w") as f:
    json.dump(analytics_data, f, indent=2)

print("dataset_analytics.json generated successfully!")
print("Overall Stats:", overall_stats)
