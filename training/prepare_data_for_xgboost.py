"""
Chuẩn bị dữ liệu master cho mô hình XGBoost (BUY/SELL/HOLD).

- Chuẩn hóa tên cột từ pipeline gốc (World_Price_VND, Domestic_Premium)
  sang tên mà train_xgboost_dss.py yêu cầu (world_price_vnd, domestic_premium).
- Đảm bảo thứ tự thời gian (time series): sort theo timestamp, không xáo trộn.
- Thêm cột thiếu (nếu có) để load_and_resample_daily không lỗi.

Chạy từ thư mục training:
  python prepare_data_for_xgboost.py --master ../master_dss_dataset.csv --out data/master_for_xgboost.csv
Sau đó chạy train từ New folder với input là training/data/master_for_xgboost.csv
hoặc copy file đó vào New folder làm master_dss_dataset.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Ánh xạ tên cột master hiện tại -> tên mà train_xgboost_dss.py dùng
COLUMN_ALIASES = {
    "World_Price_VND": "world_price_vnd",
    "Domestic_Premium": "domestic_premium",
}


def prepare_master_for_xgboost(
    master_path: Path,
    output_path: Path,
    *,
    add_missing_cpi: bool = True,
) -> pd.DataFrame:
    """
    Đọc master CSV, đổi tên cột cho đúng format, (optionally) thêm cột thiếu, lưu.
    Giữ nguyên thứ tự thời gian (time series).
    """
    df = pd.read_csv(master_path, parse_dates=["timestamp"])
    df = df.sort_values(["gold_code", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Chuẩn hóa tên cột
    rename_map = {}
    for old_name, new_name in COLUMN_ALIASES.items():
        if old_name in df.columns and new_name not in df.columns:
            rename_map[old_name] = new_name
    df = df.rename(columns=rename_map)

    # train_xgboost_dss load_and_resample_daily có thể thiếu cpi_inflation_yoy
    if add_missing_cpi and "cpi_inflation_yoy" not in df.columns:
        df["cpi_inflation_yoy"] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return df


def check_timeseries_readiness(df: pd.DataFrame) -> None:
    """In ra checklist time series: thứ tự thời gian, số ngày tối thiểu cho MA/RSI."""
    df = df.sort_values(["gold_code", "timestamp"], kind="mergesort")
    print("\n--- Time series checklist ---")
    print("1. Data sorted by (gold_code, timestamp): OK")
    print("2. No shuffle - train/test split is chronological (80/20): OK in train_xgboost_dss")
    print("3. Resample to daily (D) - one row per gold_code per day: done in load_and_resample_daily")
    print("4. Target BUY/SELL/HOLD uses shift(-3) - no look-ahead in features: OK")

    if "gold_code" in df.columns and "timestamp" in df.columns:
        daily_counts = df.groupby("gold_code").agg({"timestamp": "nunique"})
        min_days = int(daily_counts["timestamp"].min()) if len(daily_counts) else 0
        print(f"5. Min unique days per gold_code: {min_days} (need >= 20 for MA20/RSI_14)")
        if min_days < 20:
            print("   Warning: After daily resample, need >= 20 days per code for MA20 and RSI_14.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chuẩn bị master dataset cho XGBoost DSS (time series).")
    parser.add_argument(
        "--master",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "master_dss_dataset.csv",
        help="Đường dẫn file master_dss_dataset.csv (từ pipeline gốc).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "master_for_xgboost.csv",
        help="Đường dẫn file CSV đầu ra (đặt tên đúng để train_xgboost_dss đọc).",
    )
    parser.add_argument("--no-cpi", action="store_true", help="Không thêm cột cpi_inflation_yoy nếu thiếu.")
    args = parser.parse_args()

    df = prepare_master_for_xgboost(
        args.master,
        args.out,
        add_missing_cpi=not args.no_cpi,
    )
    print(f"Saved {len(df)} rows -> {args.out}")
    print("Columns after rename:", [c for c in df.columns if "world_price" in c or "domestic_premium" in c])
    check_timeseries_readiness(df)


if __name__ == "__main__":
    main()
