"""
Script 1: Load the M4 gold feature table from Iceberg via Impala.
Replaces 01_generate_data.py — real pipeline output, not synthetic data.
"""

import cml.data_v1 as cmldata

from features import (
    CONNECTION, SOURCE_TABLE, FEATURE_ORDER, LABEL, NUMERIC_FEATURES,
)

columns = ", ".join(FEATURE_ORDER + [LABEL])
query = f"SELECT {columns} FROM {SOURCE_TABLE}"

conn = cmldata.get_connection(CONNECTION)
df = conn.get_pandas_dataframe(query)

# float64 across all numerics so NaN survives into the MLflow signature.
# int64 columns cannot hold NaN, which breaks schema enforcement at serving
# time for exactly the new-to-credit records we most want to score.
df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype("float64")

df.to_csv("loan_data.csv", index=False)

print(f"Loaded {len(df):,} rows from {SOURCE_TABLE}")
print(f"Features     : {len(FEATURE_ORDER)}")
print(f"Default rate : {df[LABEL].mean():.2%}")
print(f"Nulls        : {int(df.isna().sum().sum()):,} "
      f"(bureau fields for new-to-credit borrowers — left as NaN on purpose)")