"""
Load & validate Case1A monthly data.

Saves:
 - data/processed/validation_report.csv
 - data/processed/outliers.csv

Usage:
 python Case1A_MonthlyForecast/eda/load_validate.py \
   --input ../data/raw/Case1A_MonthlyData.csv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
RAW_DIR = PROJ / "data" / "raw"
PROCESSED_DIR = PROJ / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def make_date_column(df: pd.DataFrame, year_col="YEAR", month_col="MONTH", date_col="Date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[year_col].astype(int).astype(str) + "-" + df[month_col].astype(int).astype(str).str.zfill(2) + "-01")
    df = df.sort_values([date_col]).reset_index(drop=True)
    return df


def set_datetime_index(df: pd.DataFrame, date_col="Date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df


@dataclass
class SeriesValidation:
    country: str
    product: str
    start: pd.Timestamp
    end: pd.Timestamp
    expected: int
    actual: int
    missing_count: int
    duplicates: int
    missing_dates: list


def validate_time_series(df: pd.DataFrame, id_cols=("Country", "Product"), value_col="Value") -> Tuple[pd.DataFrame, pd.DataFrame]:
    reports = []
    outlier_rows = []

    for (country, product), grp in df.groupby(list(id_cols)):
        g = grp.sort_index()
        start, end = g.index.min(), g.index.max()

        # expected number of monthly periods
        expected = (end.year - start.year) * 12 + (end.month - start.month) + 1  # see math below
        actual = g.shape[0]

        # duplicates on the index
        dup_count = g.index.duplicated().sum()

        # expected full monthly index and missing dates
        full = pd.date_range(start=start, end=end, freq="MS")
        missing = sorted(list(set(full) - set(g.index)))
        missing_count = len(missing)

        reports.append(
            SeriesValidation(
                country=country,
                product=product,
                start=start,
                end=end,
                expected=expected,
                actual=actual,
                missing_count=missing_count,
                duplicates=int(dup_count),
                missing_dates=[d.strftime("%Y-%m") for d in missing],
            )
        )

        # Outlier detection (IQR per series)
        if value_col in g.columns:
            vals = g[value_col].dropna()
            if not vals.empty:
                q1 = vals.quantile(0.25)
                q3 = vals.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask_out = (g[value_col] < lower) | (g[value_col] > upper)
                if mask_out.any():
                    tmp = g.loc[mask_out, :].reset_index()
                    tmp["Country"] = country
                    tmp["Product"] = product
                    tmp["outlier_lower"] = lower
                    tmp["outlier_upper"] = upper
                    outlier_rows.append(tmp)

    report_df = pd.DataFrame([r.__dict__ for r in reports])
    if outlier_rows:
        outliers_df = pd.concat(outlier_rows, ignore_index=True)
    else:
        outliers_df = pd.DataFrame(columns=list(df.reset_index().columns) + ["outlier_lower", "outlier_upper", "Country", "Product"])

    return report_df, outliers_df


def main(input_path: str):
    input_path = Path(input_path)
    df = load_csv(input_path)
    df = make_date_column(df, year_col="YEAR", month_col="MONTH", date_col="Date")

    # basic checks
    if "Country" not in df.columns or "Product" not in df.columns or "Value" not in df.columns:
        raise KeyError("Expected columns: YEAR, MONTH, Country, Product, Value")

    df_idx = set_datetime_index(df, date_col="Date")

    # duplicates in raw rows (non-index)
    dup_rows = df_idx.reset_index().duplicated(subset=["Date", "Country", "Product"], keep=False).sum()

    report_df, outliers_df = validate_time_series(df_idx, id_cols=("Country", "Product"), value_col="Value")

    # add global summary
    summary = {
        "total_rows": int(df.shape[0]),
        "unique_country_product": int(df.set_index(["Country", "Product"]).index.nunique()),
        "duplicate_rows_on_keys": int(dup_rows),
    }
    report_df_path = PROCESSED_DIR / "validation_report.csv"
    outliers_path = PROCESSED_DIR / "outliers.csv"

    report_df.to_csv(report_df_path, index=False)
    outliers_df.to_csv(outliers_path, index=False)

    print("Validation summary:")
    print(summary)
    print(f"Per-series report written to: {report_df_path}")
    print(f"Outliers written to: {outliers_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default=RAW_DIR / "case1a.csv", help="input csv path")
    args = ap.parse_args()
    main(args.input)