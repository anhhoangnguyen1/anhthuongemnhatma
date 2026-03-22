from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Giá thế giới trong CSV là USD/lượng tây (troy oz) chuyển sang VND/lượng ta (37.5g) — cùng công thức dữ liệu cũ.
_GRAMS_PER_TROY_OZ = 31.1034768
_GRAMS_PER_LUONG_TA = 37.5
_USD_OZ_TO_VND_LUONG = _GRAMS_PER_LUONG_TA / _GRAMS_PER_TROY_OZ


def _to_numeric_col(s: pd.Series) -> pd.Series:
    """Chuỗi kiểu '5,327.42' -> float; đã số thì giữ."""
    if s.dtype == object:
        s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _broadcast_median_by_date(df: pd.DataFrame, cols: list[str], ts: pd.Series) -> None:
    """Các cột vĩ mô thường giống nhau trong ngày — lấp NaN bằng median theo ngày."""
    day = ts.dt.normalize()
    for c in cols:
        if c not in df.columns:
            continue
        med = df.groupby(day, sort=False)[c].transform("median")
        df[c] = df[c].fillna(med)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.rolling(period, min_periods=period).mean()
    avg_down = down.rolling(period, min_periods=period).mean().replace(0, np.nan)
    rs = avg_up / avg_down
    return 100 - (100 / (1 + rs))


def main() -> None:
    p = Path(__file__).resolve().parents[1] / "training" / "master_dss_dataset.csv"
    df = pd.read_csv(p, low_memory=False)

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    start = pd.Timestamp("2026-03-02")
    sell = _to_numeric_col(df["sell_price"]) if "sell_price" in df.columns else pd.Series(np.nan, index=df.index)
    wv = _to_numeric_col(df["World_Price_VND"]) if "World_Price_VND" in df.columns else pd.Series(np.nan, index=df.index)
    wusd = _to_numeric_col(df["World_Price_USD_Ounce"]) if "World_Price_USD_Ounce" in df.columns else pd.Series(np.nan, index=df.index)
    prem = _to_numeric_col(df["Domestic_Premium"]) if "Domestic_Premium" in df.columns else pd.Series(np.nan, index=df.index)
    fake_mask = (ts >= start) & wusd.isna() & wv.eq(sell) & prem.fillna(0).eq(0)
    n_fake = int(fake_mask.sum())
    df.loc[fake_mask, "World_Price_VND"] = np.nan
    df.loc[fake_mask, "Domestic_Premium"] = np.nan

    fill_cols = [
        "World_Price_USD_Ounce",
        "World_Price_VND",
        "usd_vnd_rate",
        "dxy_index",
        "fed_rate",
        "interest_rate_state",
        "interest_rate_market",
        "interest_rate_spread",
    ]
    for c in fill_cols:
        if c in df.columns:
            df[c] = _to_numeric_col(df[c])

    # Phải sort theo datetime — sort chuỗi M/D/YYYY sai thứ tự (vd: 3/10 trước 3/7).
    df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["gold_code", "_ts"], na_position="last").reset_index(drop=True)
    ts = df["_ts"]
    grp = df.groupby("gold_code", dropna=False)
    for c in fill_cols:
        if c in df.columns:
            df[c] = grp[c].ffill().bfill()

    # Mã mới / ngày mới: lấp macro từ các dòng khác cùng ngày
    _broadcast_median_by_date(df, fill_cols, ts)

    # Suy ra giá thế giới VND/lượng từ USD/oz + tỷ giá (khi thiếu World_Price_VND)
    if {"World_Price_USD_Ounce", "usd_vnd_rate", "World_Price_VND"}.issubset(df.columns):
        need_vnd = df["World_Price_VND"].isna() & df["World_Price_USD_Ounce"].notna() & df["usd_vnd_rate"].notna()
        df.loc[need_vnd, "World_Price_VND"] = (
            df.loc[need_vnd, "World_Price_USD_Ounce"].astype(float)
            * df.loc[need_vnd, "usd_vnd_rate"].astype(float)
            * _USD_OZ_TO_VND_LUONG
        )

    if {"sell_price", "World_Price_VND"}.issubset(df.columns):
        df["Domestic_Premium"] = _to_numeric_col(df["sell_price"]) - _to_numeric_col(df["World_Price_VND"])

    df["MA_7"] = grp["sell_price"].transform(
        lambda x: pd.to_numeric(x, errors="coerce").rolling(7, min_periods=3).mean()
    )
    df["MA_30"] = grp["sell_price"].transform(
        lambda x: pd.to_numeric(x, errors="coerce").rolling(30, min_periods=10).mean()
    )
    df["RSI_14"] = grp["sell_price"].transform(rsi)
    sell_num = _to_numeric_col(df["sell_price"])
    df["Momentum_1D_Pct"] = grp["sell_price"].transform(
        lambda x: pd.to_numeric(x, errors="coerce").pct_change(1) * 100
    )
    df["Momentum_3D_Pct"] = grp["sell_price"].transform(
        lambda x: pd.to_numeric(x, errors="coerce").pct_change(3) * 100
    )
    df["Momentum_7D_Pct"] = grp["sell_price"].transform(
        lambda x: pd.to_numeric(x, errors="coerce").pct_change(7) * 100
    )
    horizon = 7
    df["Target_Price_Next_Shift"] = grp["sell_price"].transform(
        lambda x: pd.to_numeric(x, errors="coerce").shift(-horizon)
    )
    future_ret = (pd.to_numeric(df["Target_Price_Next_Shift"], errors="coerce") - sell_num) / sell_num * 100
    # Ngưỡng giữ đồng nhất với train mặc định buy_pct=0.5
    df["Target_Trend"] = np.where(
        future_ret.isna(),
        pd.NA,
        np.where(future_ret > 0.5, "BUY", "NOT_BUY"),
    )

    df = df.sort_values(["_ts", "gold_code"], na_position="last").drop(columns=["_ts"]).reset_index(drop=True)
    df.to_csv(p, index=False)

    chk = df[pd.to_datetime(df["timestamp"], errors="coerce") >= start]
    cols = [
        "World_Price_USD_Ounce",
        "World_Price_VND",
        "Domestic_Premium",
        "MA_7",
        "MA_30",
        "RSI_14",
        "Momentum_1D_Pct",
        "Momentum_3D_Pct",
        "Momentum_7D_Pct",
        "Target_Price_Next_Shift",
        "Target_Trend",
    ]
    print("fake_reset", n_fake)
    print(chk[cols].isna().sum().to_string())


if __name__ == "__main__":
    main()
