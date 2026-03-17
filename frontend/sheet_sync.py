"""
Đồng bộ dữ liệu từ Google Sheet realtime vào master_dss_dataset.csv.
Chạy khi khởi động app (python app.py): nếu có GOOGLE_SHEET_ID và credentials,
sẽ kéo dữ liệu mới từ sheet và append vào file master để giao diện dùng realtime.

Cột sheet: timestamp, gold_code, buy_price, sell_price, usd_vnd_rate, fed_rate,
           cpi_inflation_yoy, dxy_index, interest_rate_state, interest_rate_market
Số có thể dạng EU: 25813,17 (dấu phẩy thập phân).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

_LOG_PREFIX = "[Sheet sync]"


def _log(msg: str) -> None:
    print(f"{_LOG_PREFIX} {msg}")


def _normalize_number(val: str | float | int) -> float | None:
    """Chuẩn hóa số: '25813,17' hoặc '3,75' -> float."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(",", ".")
    s = re.sub(r"\s+", "", s)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_date(val: str) -> str | None:
    """Chuẩn hóa ngày: '2026-02-11' giữ nguyên; hoặc dạng khác -> ISO."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    # Đã là YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return s[:10]
    try:
        parsed = pd.to_datetime(s)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return None


def fetch_sheet_as_dataframe(sheet_id: str, credentials_path: str | None = None) -> pd.DataFrame | None:
    """
    Đọc toàn bộ sheet (sheet đầu tiên) thành DataFrame.
    Cột: timestamp, gold_code, buy_price, sell_price, usd_vnd_rate, fed_rate,
         cpi_inflation_yoy, dxy_index, interest_rate_state, interest_rate_market.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as e:
        _log(f"Thiếu thư viện: gspread hoặc google-auth. pip install gspread google-auth — {e}")
        return None

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    raw_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not raw_path:
        _log("GOOGLE_APPLICATION_CREDENTIALS chưa cấu hình trong .env")
        return None
    # Resolve path: nếu tương đối thì xét từ thư mục gốc project (parent của frontend)
    creds_path = Path(raw_path)
    if not creds_path.is_absolute():
        root = Path(__file__).resolve().parent.parent
        creds_path = root / raw_path
    if not creds_path.exists():
        _log(f"Không tìm thấy file credentials: {creds_path}")
        return None

    try:
        # Chỉ hỗ trợ Service Account JSON (file có "type": "service_account").
        with open(creds_path, encoding="utf-8") as f:
            import json as _json
            data = _json.load(f)
        if data.get("type") != "service_account":
            _log(f"File {creds_path.name} không phải Service Account key (type={data.get('type')}). Cần tạo key JSON từ GCP Service Account.")
            return None
        creds = Credentials.from_service_account_file(str(creds_path), scopes=scopes)
        gc = gspread.authorize(creds)
        workbook = gc.open_by_key(sheet_id)
        sheet = workbook.sheet1
        _log(f"Đang đọc sheet: '{sheet.title}' (sheet đầu tiên, gid có thể khác nếu data ở tab khác)")
        rows = sheet.get_all_values()
    except Exception as e:
        _log(f"Lỗi đọc Google Sheet: {e}")
        return None

    if not rows or len(rows) < 2:
        _log("Sheet trống hoặc chỉ có 1 dòng (header). Cần ít nhất header + 1 dòng dữ liệu.")
        return pd.DataFrame()

    headers = [str(h).strip().lower().replace(" ", "_") for h in rows[0]]
    _log(f"Cột sheet (sau chuẩn hóa): {headers[:6]}..." if len(headers) > 6 else f"Cột sheet: {headers}")
    # Map tên cột thường gặp
    col_map = {
        "timestamp": "timestamp",
        "gold_code": "gold_code",
        "buy_price": "buy_price",
        "sell_price": "sell_price",
        "usd_vnd_rate": "usd_vnd_rate",
        "fed_rate": "fed_rate",
        "cpi_inflation_yoy": "cpi_inflation_yoy",
        "dxy_index": "dxy_index",
        "interest_rate_state": "interest_rate_state",
        "interest_rate_market": "interest_rate_market",
    }
    data = []
    for row in rows[1:]:
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        raw = dict(zip(headers, row[: len(headers)]))
        ts = _normalize_date(raw.get("timestamp") or raw.get("date"))
        gold = (raw.get("gold_code") or "").strip()
        if not ts or not gold:
            continue
        buy = _normalize_number(raw.get("buy_price"))
        sell = _normalize_number(raw.get("sell_price"))
        usd_vnd = _normalize_number(raw.get("usd_vnd_rate"))
        fed = _normalize_number(raw.get("fed_rate"))
        cpi = _normalize_number(raw.get("cpi_inflation_yoy"))
        dxy = _normalize_number(raw.get("dxy_index"))
        ir_state = _normalize_number(raw.get("interest_rate_state"))
        ir_market = _normalize_number(raw.get("interest_rate_market"))
        if sell is None:
            continue
        # Bắt buộc có sell_price; buy có thể dùng sell nếu thiếu
        if buy is None:
            buy = sell
        row_out = {
            "timestamp": ts,
            "gold_code": gold,
            "buy_price": buy,
            "sell_price": sell,
            "usd_vnd_rate": usd_vnd if usd_vnd is not None else 0.0,
            "fed_rate": fed if fed is not None else 0.0,
            "cpi_inflation_yoy": cpi,
            "dxy_index": dxy if dxy is not None else 0.0,
            "interest_rate_state": ir_state if ir_state is not None else 0.0,
            "interest_rate_market": ir_market if ir_market is not None else 0.0,
        }
        data.append(row_out)

    if not data:
        _log("Sau khi parse, không có dòng nào hợp lệ (cần timestamp + gold_code + sell_price).")
        return pd.DataFrame()
    df = pd.DataFrame(data)
    ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
    _log(f"Sheet: đọc được {len(df)} dòng, khoảng ngày {ts_min} → {ts_max}, {df['gold_code'].nunique()} mã vàng.")
    return df


def sync_master_from_google_sheet(
    master_path: Path,
    sheet_id: str | None = None,
    credentials_path: str | None = None,
) -> int | None:
    """
    Lấy dữ liệu mới từ Google Sheet, merge vào file master (append theo timestamp+gold_code).
    Trả về số dòng mới được thêm, hoặc None nếu bỏ qua/lỗi.
    """
    sheet_id = sheet_id or os.environ.get("GOOGLE_SHEET_ID")
    if not sheet_id:
        _log("GOOGLE_SHEET_ID chưa cấu hình trong .env — bỏ qua sync.")
        return None
    credentials_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    _log(f"Bắt đầu sync: sheet_id={sheet_id[:8]}..., master={master_path}")

    df_sheet = fetch_sheet_as_dataframe(sheet_id, credentials_path)
    if df_sheet is None or df_sheet.empty:
        _log("Không lấy được dữ liệu từ Sheet (fetch trả về None hoặc rỗng).")
        return None

    # Thêm cột master cần mà sheet không có (pipeline dùng world_price_vnd, domestic_premium)
    df_sheet["interest_rate_spread"] = (
        df_sheet["interest_rate_market"].fillna(0) - df_sheet["interest_rate_state"].fillna(0)
    )
    df_sheet["World_Price_VND"] = df_sheet["sell_price"].values  # placeholder: coi premium = 0
    df_sheet["Domestic_Premium"] = 0.0

    # Đọc master hiện tại
    if not master_path.exists():
        _log(f"File master chưa tồn tại — ghi toàn bộ {len(df_sheet)} dòng từ Sheet vào {master_path}")
        df_sheet.to_csv(master_path, index=False, date_format="%Y-%m-%d")
        return len(df_sheet)

    try:
        df_master = pd.read_csv(master_path, nrows=1000000)
    except Exception as e:
        _log(f"Lỗi đọc file master: {e}")
        return None

    # Cột key để loại trùng
    key_cols = ["timestamp", "gold_code"]
    for c in key_cols:
        if c not in df_sheet.columns:
            _log(f"Sheet thiếu cột bắt buộc: {c}")
            return None
    df_sheet["timestamp"] = pd.to_datetime(df_sheet["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_master["timestamp"] = pd.to_datetime(df_master["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")

    master_dates = df_master["timestamp"].dropna().unique()
    master_min, master_max = (min(master_dates), max(master_dates)) if len(master_dates) else (None, None)
    _log(f"Master: {len(df_master)} dòng, ngày {master_min} → {master_max}")

    # Chỉ giữ dòng từ sheet mà (timestamp, gold_code) chưa có trong master
    existing_keys = set(zip(df_master["timestamp"].astype(str), df_master["gold_code"].astype(str)))
    df_sheet["_key"] = list(zip(df_sheet["timestamp"].astype(str), df_sheet["gold_code"].astype(str)))
    new_keys = set(df_sheet["_key"]) - existing_keys
    if not new_keys:
        df_sheet = df_sheet.drop(columns=["_key"])
        _log("Không có dòng mới: mọi (timestamp, gold_code) trên Sheet đã có trong master.")
        return 0

    df_new = df_sheet[df_sheet["_key"].isin(new_keys)].drop(columns=["_key"]).copy()
    if df_new.empty:
        _log("Sau khi lọc, không còn dòng mới.")
        return 0
    new_dates = df_new["timestamp"].dropna().unique()
    new_min, new_max = (min(new_dates), max(new_dates)) if len(new_dates) else (None, None)
    _log(f"Có {len(df_new)} dòng mới (ngày {new_min} → {new_max}), đang ghi vào master...")

    # Khớp cột với master: cùng thứ tự và tên cột, thiếu thì NaN
    for c in df_master.columns:
        if c not in df_new.columns:
            df_new[c] = None
    df_new = df_new.reindex(columns=df_master.columns, fill_value=None)
    df_merged = pd.concat([df_master, df_new], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=key_cols, keep="last")
    df_merged = df_merged.sort_values(key_cols).reset_index(drop=True)
    try:
        df_merged.to_csv(master_path, index=False, date_format="%Y-%m-%d")
    except Exception as e:
        _log(f"Lỗi ghi file master: {e}")
        return None
    _log(f"Đã ghi xong: thêm {len(df_new)} dòng, master hiện có {len(df_merged)} dòng.")
    return len(df_new)
