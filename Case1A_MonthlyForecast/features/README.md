# Features

Scripts to generate feature tables for Case1A_MonthlyForecast. Place feature engineering scripts here and write outputs to `../data/processed/`.

Usage
-----

The repository includes a feature generation script `feature_engineering.py` which builds time-based, lag, rolling, cyclical and categorical features and writes a CSV feature table to `../data/processed/features_table.csv` by default.

Run examples:

```bash
# run using the Python interpreter (no need to change permissions)
python3 features/feature_engineering.py \
	--input ../data/raw/Case1A_MonthlyData.csv \
	--output ../data/processed/features_table.csv \
	--force

# or execute directly (script contains a shebang and must be executable):
./features/feature_engineering.py --input ../data/raw/Case1A_MonthlyData.csv --output ../data/processed/features_table.csv --force
```

Caveat: duplicate rows per month
--------------------------------

Depending on the raw data, you may have multiple rows for the same (Country, Product, YEAR, MONTH). In that case the script will generate features for each input row and you may see duplicated timestamp rows in the resulting feature table (for example `Value_x`/`Value_y` columns from merges).

Recommended approaches:

- Aggregate the raw monthly values to a single value per (Country, Product, date) before feature generation. Common choices are `mean`, `sum` or `last` depending on your business semantics. Example (to create a mean-aggregated file):

```python
import pandas as pd
df = pd.read_csv('data/raw/Case1A_MonthlyData.csv')
df['date'] = pd.to_datetime(dict(year=df.YEAR, month=df.MONTH, day=1))
agg = df.groupby(['Country','Product','date'], as_index=False)['Value'].mean()
agg.to_csv('data/raw/Case1A_MonthlyData_agg.csv', index=False)
```

Then run the feature script pointing at the aggregated file.

- Alternatively, we can add an aggregation step to `feature_engineering.py` and expose a CLI flag (recommended). If you want that, request "Add aggregation CLI option" and I'll implement it.

Notes
-----
- You can run the script without changing file permissions by calling `python3 features/feature_engineering.py ...`.
- If you prefer executing the script directly (`./features/feature_engineering.py`), ensure it has a shebang and is executable (`chmod +x features/feature_engineering.py`).
