"""
Tải dữ liệu từ nguồn public (XAUUSD, Fed, DXY, USD/VND) và lưu đúng format 5 file
để chạy prepare_gold_dss_pipeline.py (xem mục 0 trong NGUON_DU_LIEU_MO_RONG.md).

Chạy: python download_public_data.py
Output: thư mục downloaded_data/ với các file CSV (đặt vào thư mục gốc hoặc dùng --input-dir).
Lãi suất VN: không có nguồn free dài hạn → giữ file interest_rate.csv hiện tại.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Can not import yfinance. Run: pip install yfinance")

import pandas as pd

try:
    import pandas_datareader as pdr
    HAS_PDR = True
except ImportError:
    HAS_PDR = False

# Thư mục output (trong thư mục gốc dự án)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "downloaded_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Số năm lấy (càng nhiều càng tốt, tùy nguồn)
YEARS_BACK = 2


def _parse_fred_date(s: str):
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return s


def download_xauusd() -> pd.DataFrame:
    """XAUUSD (vàng thế giới) từ yfinance GC=F. Trả về DataFrame có timestamp, gold_code, buy_price, sell_price."""
    end = datetime.now()
    start = end - timedelta(days=YEARS_BACK * 365)
    df = yf.download("GC=F", start=start, end=end, progress=False, auto_adjust=True)
    if df.empty or not hasattr(df, "columns"):
        return pd.DataFrame()
    df = df.reset_index()
    # yfinance có thể trả MultiIndex (Price, Ticker) -> lấy tên cột đầu (Open, Close, ...)
    def _col_name(c):
        if isinstance(c, tuple):
            return str(c[0]).lower()
        return str(c).lower()
    cols = [_col_name(c) for c in df.columns]
    df.columns = cols
    # Open/Close -> buy/sell; pipeline có thể dùng close cho cả hai nếu thiếu
    if "open" in df.columns and "close" in df.columns:
        df["buy_price"] = df["open"].astype(float)
        df["sell_price"] = df["close"].astype(float)
    elif "close" in df.columns:
        df["buy_price"] = df["close"].astype(float)
        df["sell_price"] = df["close"].astype(float)
    else:
        return pd.DataFrame()
    df["gold_code"] = "XAUUSD"
    if "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        df["timestamp"] = df.index.astype(str)
    out = df[["timestamp", "gold_code", "buy_price", "sell_price"]].copy()
    out = out.dropna(subset=["buy_price", "sell_price"])
    return out


def download_usd_vnd() -> pd.DataFrame:
    """USD/VND từ yfinance. Lưu timestamp, usd_vnd_rate."""
    end = datetime.now()
    start = end - timedelta(days=YEARS_BACK * 365)
    # VND=X: có thể là USD/VND (số VND cho 1 USD)
    df = yf.download("VND=X", start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    def _col_name(c):
        return str(c[0]).lower() if isinstance(c, tuple) else str(c).lower()
    df.columns = [_col_name(c) for c in df.columns]
    if "close" not in df.columns:
        return pd.DataFrame()
    df["usd_vnd_rate"] = df["close"].astype(float)
    if "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        df["datetime"] = df.index.astype(str)
    out = df[["datetime", "usd_vnd_rate"]].copy()
    out = out.rename(columns={"datetime": "timestamp"})
    return out.dropna(subset=["usd_vnd_rate"])


def download_fred_via_pdr(series_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Tải FRED qua pandas_datareader (không cần API key)."""
    if not HAS_PDR:
        return pd.DataFrame()
    try:
        df = pdr.get_data_fred(series_id, start=start, end=end)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    if "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    col_val = [c for c in df.columns if c not in ("date", "timestamp")][:1]
    if not col_val:
        return pd.DataFrame()
    df = df.rename(columns={col_val[0]: series_id.lower()})
    return df[["timestamp", series_id.lower()]].dropna()


def download_fred_series_txt(series_id: str) -> pd.DataFrame:
    """Tải 1 series từ FRED file .txt (fallback)."""
    url = f"https://fred.stlouisfed.org/data/{series_id}.txt"
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=15) as r:
            raw = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return pd.DataFrame()
    lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.strip().startswith("DATE") and re.match(r"^\d{4}-\d{2}-\d{2}", l.strip())]
    if not lines:
        return pd.DataFrame()
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                date_str = parts[0]
                val = float(parts[1])
                rows.append((date_str + " 00:00:00", val))
            except ValueError:
                continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["timestamp", series_id.lower()])


def download_fed() -> pd.DataFrame:
    """Fed funds rate: thử pandas_datareader FRED trước, rồi fallback .txt."""
    end = datetime.now()
    start = end - timedelta(days=YEARS_BACK * 365)
    df = download_fred_via_pdr("FEDFUNDS", start, end)
    if not df.empty:
        df = df.rename(columns={"fedfunds": "fed_rate"})
        return df
    df = download_fred_series_txt("FEDFUNDS")
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"fedfunds": "fed_rate"})
    return df


def download_dxy() -> pd.DataFrame:
    """DXY: thử FRED qua pdr (DTWEXBGS), rồi .txt, rồi yfinance."""
    end = datetime.now()
    start = end - timedelta(days=YEARS_BACK * 365)
    df = download_fred_via_pdr("DTWEXBGS", start, end)
    if not df.empty:
        df = df.rename(columns={"dtwexbgs": "dxy_index"})
        return df
    df = download_fred_series_txt("DTWEXBGS")
    if not df.empty:
        df = df.rename(columns={"dtwexbgs": "dxy_index"})
        return df
    # Fallback: yfinance DX-Y.NYB
    end = datetime.now()
    start = end - timedelta(days=YEARS_BACK * 365)
    try:
        d = yf.download("DX-Y.NYB", start=start, end=end, progress=False, auto_adjust=True)
    except Exception:
        return pd.DataFrame()
    if d.empty:
        return pd.DataFrame()
    d = d.reset_index()
    def _col_name(c):
        return str(c[0]).lower() if isinstance(c, tuple) else str(c).lower()
    d.columns = [_col_name(c) for c in d.columns]
    if "close" not in d.columns:
        return pd.DataFrame()
    d["timestamp"] = pd.to_datetime(d.get("date", d.index)).dt.strftime("%Y-%m-%d %H:%M:%S")
    d["dxy_index"] = d["close"].astype(float)
    return d[["timestamp", "dxy_index"]].dropna()


def try_vang_today() -> pd.DataFrame:
    """Thử lấy giá vàng VN từ vang.today API (tối đa 30 ngày). Trả về DataFrame có timestamp, gold_code, buy_price, sell_price."""
    try:
        import urllib.request
        # Mã SJC 9999
        url = "https://api.vang.today/api/prices?type=SJL1L10&days=30"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = r.read().decode("utf-8")
    except Exception:
        return pd.DataFrame()
    try:
        import json
        j = json.loads(data)
    except Exception:
        return pd.DataFrame()
    rows = []
    for item in j if isinstance(j, list) else (j.get("data", j) or []):
        if isinstance(item, dict):
            ts = item.get("updated_at") or item.get("timestamp") or item.get("date")
            buy = item.get("buy") or item.get("gia_mua") or item.get("buy_price")
            sell = item.get("sell") or item.get("gia_ban") or item.get("sell_price")
            if ts and (buy is not None or sell is not None):
                rows.append({
                    "timestamp": ts,
                    "gold_code": item.get("gold_code", "SJL1L10"),
                    "buy_price": float(buy) if buy is not None else float(sell),
                    "sell_price": float(sell) if sell is not None else float(buy),
                })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main():
    print("Output folder:", OUTPUT_DIR)
    print()

    # 1. XAUUSD
    print("1. Downloading XAUUSD (GC=F)...")
    xau = download_xauusd()
    if xau.empty:
        print("   Failed to get XAUUSD.")
    else:
        path = OUTPUT_DIR / "xauusd.csv"
        xau.to_csv(path, index=False, encoding="utf-8-sig")
        print("   Saved", len(xau), "rows ->", path)

    # 2. Vàng VN (thử API)
    print("2. Trying vang.today API (VN gold, 30 days)...")
    vn_gold = try_vang_today()
    if not vn_gold.empty:
        path = OUTPUT_DIR / "gold_vn_30d.csv"
        vn_gold.to_csv(path, index=False, encoding="utf-8-sig")
        print("   Saved", len(vn_gold), "rows ->", path)
    else:
        print("   No data (API limit or error). Keep your existing GOLD_PRICE.csv for domestic codes.")

    # 3. Gộp vàng: XAUUSD + (nếu có) VN 30d; nếu có file gốc GOLD_PRICE thì gộp thêm vàng nội địa
    if not xau.empty:
        gold_combined = xau.copy()
        if not vn_gold.empty:
            gold_combined = pd.concat([gold_combined, vn_gold], ignore_index=True)
        # Gộp vàng nội địa từ project root (nếu có) để pipeline có đủ mã VN
        root_gold = PROJECT_ROOT / "GOLD_PRICE.csv"
        if root_gold.exists():
            try:
                for enc in ("utf-8-sig", "utf-8", "cp1258"):
                    try:
                        rd = pd.read_csv(root_gold, encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    rd = pd.read_csv(root_gold)
                cols = [str(c) for c in rd.columns]
                def _norm(s):
                    return s.lower().replace("á", "a").replace("à", "a").replace("ã", "a").replace("ạ", "a").replace("ă", "a").replace("â", "a").replace("é", "e").replace("è", "e").replace("ê", "e").replace("í", "i").replace("ó", "o").replace("ô", "o").replace("ơ", "o").replace("ú", "u").replace("ư", "u").replace("đ", "d")
                ts_col = next((c for c in cols if "thoi" in _norm(c) or "cap_nhat" in _norm(c) or "datetime" in _norm(c) or "timestamp" in _norm(c) or "ngay" in _norm(c)), None)
                code_col = next((c for c in cols if "vang" in _norm(c) or "code" in _norm(c) or "ma" in _norm(c)), None)
                buy_col = next((c for c in cols if "mua" in _norm(c) or "buy" in _norm(c)), None)
                sell_col = next((c for c in cols if "ban" in _norm(c) or "sell" in _norm(c)), None)
                if ts_col and code_col and buy_col and sell_col:
                    rd = rd[[ts_col, code_col, buy_col, sell_col]].copy()
                    rd = rd.rename(columns={ts_col: "timestamp", code_col: "gold_code", buy_col: "buy_price", sell_col: "sell_price"})
                    rd["gold_code"] = rd["gold_code"].astype(str).str.strip().str.upper()
                    domestic = rd[rd["gold_code"] != "XAUUSD"].dropna(subset=["buy_price", "sell_price"])
                    if not domestic.empty:
                        gold_combined = pd.concat([gold_combined, domestic], ignore_index=True)
                        print("   Merged domestic gold from root GOLD_PRICE.csv ->", len(domestic), "rows")
            except Exception as e:
                print("   Could not merge root GOLD_PRICE:", e)
        path = OUTPUT_DIR / "GOLD_PRICE.csv"
        gold_combined.to_csv(path, index=False, encoding="utf-8-sig")
        print("   Combined gold ->", path)
    else:
        print("   Skipped combined gold (no XAUUSD).")

    # 4. USD/VND
    print("3. Downloading USD/VND (VND=X)...")
    usd = download_usd_vnd()
    if usd.empty:
        print("   Failed. Keep your existing usd_vnd_rate_live.csv.")
    else:
        path = OUTPUT_DIR / "usd_vnd_rate_live.csv"
        usd.to_csv(path, index=False, encoding="utf-8-sig")
        print("   Saved", len(usd), "rows ->", path)

    # 5. Fed
    print("4. Downloading Fed rate (FRED FEDFUNDS)...")
    fed = download_fed()
    if fed.empty:
        print("   Failed. Keep your existing fed_rate_live.csv.")
    else:
        path = OUTPUT_DIR / "fed_rate_live.csv"
        fed.to_csv(path, index=False, encoding="utf-8-sig")
        print("   Saved", len(fed), "rows ->", path)

    # 6. DXY
    print("5. Downloading DXY...")
    dxy = download_dxy()
    if dxy.empty:
        print("   Failed. Keep your existing dxy_history.csv.")
    else:
        path = OUTPUT_DIR / "dxy_history.csv"
        dxy.to_csv(path, index=False, encoding="utf-8-sig")
        print("   Saved", len(dxy), "rows ->", path)

    print()
    print("6. Interest rate VN: no free daily source. Copy your existing interest_rate.csv to downloaded_data/ if you run pipeline on that folder.")
    print()
    print("Done. Fed + DXY from FRED/yfinance (new source). To run pipeline:")
    print("  - Copy interest_rate.csv to downloaded_data/ if not already.")
    print("  - From project root: python prepare_gold_dss_pipeline.py --input-dir downloaded_data --output-file master_dss_dataset.csv")
    print("    (output will be: downloaded_data/master_dss_dataset.csv)")
    print("  Or copy all 5 files from downloaded_data/ to project root to replace (backup originals first).")


if __name__ == "__main__":
    main()
