# Cấu trúc dự án – Dự đoán giá vàng DSS

## Luồng chính: Training → Pipeline → Frontend

```
input_1year/                 prepare_gold_dss_pipeline.py     New folder/master_dss_dataset.csv
(CSV nguồn duy nhất)         (merge macro, world, premium)              |
                                                                         v
                                                    New folder/train_xgboost_dss.py
                                                    (feature + target → XGBoost)
                                                                         |
                                                                         v
                                                    New folder/output/
                                                                         |
                                                                         v
frontend/app.py  <-- đọc master + output/*  -->  /api/predict, /api/macro, /api/date-detail
```

## Thư mục và file ở root

- **prepare_gold_dss_pipeline.py** – Pipeline build master (mặc định đọc `input_1year/`, ghi ra file chỉ định).
- **.env**, **.gitignore**, **README.md**, **PROJECT_STRUCTURE.md**
- **GOLD_PRICE.csv**, **GOLD_PRICE_1_YEAR_SCRAPED.csv**, **MACRO_FEATURES_1_YEAR.csv** – Dữ liệu nguồn (output của scripts hoặc thu thập tay).
- **input_1year/** – Thư mục input duy nhất cho pipeline: `GOLD_PRICE.csv`, `usd_vnd_rate_live.csv`, `interest_rate.csv`, `dxy_history.csv`, `fed_rate_live.csv`. Tuỳ chọn: `NEWS_SENTIMENT.csv`, `NEWS_IMPACT_DAILY.csv` (đặt trong đây nếu dùng).
- **scripts/** – Scripts thu thập/chuẩn hoá dữ liệu (chạy từ root: `python scripts/xxx.py`):
  - **fetch_macro_1_year.py** – Macro + scrape vàng → `MACRO_FEATURES_1_YEAR.csv`, `GOLD_PRICE_1_YEAR_SCRAPED.csv`, cache.
  - **build_master_1year.py** – Từ scraped + macro → ghi vào `input_1year/`.
  - **fetch_news_sentiment_marketaux.py** – Tạo `NEWS_SENTIMENT.csv` (tuỳ chọn; đặt vào `input_1year/` nếu dùng).
  - **assess_news_impact_llm.py** – Tạo `NEWS_IMPACT_DAILY.csv` (tuỳ chọn; đặt vào `input_1year/` nếu dùng).
- **New folder/** – Training + master + output (xem dưới).
- **frontend/** – Flask app + LLM + templates.

### Training & model (New folder/)
- **master_dss_dataset.csv** – Master duy nhất cho train và predict.
- **train_xgboost_dss.py** – Train XGBoost.
- **preprocessing.py** – Tiền xử lý (Winsorizer).
- **output/** – `xgboost_dss_model.pkl`, `scaler_xgboost_dss.pkl`, `label_encoder_dss.pkl`, `model_config.json`, `evaluation_metrics.txt`, `xgboost_feature_importance.png`.

### Frontend
- **frontend/app.py** – API predict, macro, date-detail; đọc master từ `New folder/master_dss_dataset.csv`, model từ `New folder/output/`. Khi chạy `python frontend/app.py`, gọi **frontend/sheet_sync.py** để kéo dữ liệu mới từ Google Sheet (nếu có GOOGLE_SHEET_ID) và append vào master.
- **frontend/sheet_sync.py** – Đồng bộ Google Sheet realtime → `New folder/master_dss_dataset.csv` (cột: timestamp, gold_code, buy_price, sell_price, usd_vnd_rate, fed_rate, cpi_inflation_yoy, dxy_index, interest_rate_state, interest_rate_market; số có thể dạng EU `25813,17`).
- **frontend/llm_adjust.py** – GNews + OpenAI: đánh giá LLM, dự đoán bổ sung theo ngày.
- **frontend/templates/index.html** – Giao diện chart, chọn mã vàng, chi tiết theo ngày.

### Cấu hình
- **.env** – OPENAI_API_KEY, GNEWS_API_KEY; tuỳ chọn: GOOGLE_SHEET_ID, GOOGLE_APPLICATION_CREDENTIALS (để sync Sheet). Cần cài `gspread` và `google-auth` nếu dùng sync Sheet.
- **.env.example** – Mẫu biến môi trường.

## Đã xoá / không dùng
- **input/** – Đã gộp dùng một nguồn: **input_1year/**.
- **master_dss_dataset.csv** (root) – Trùng; app chỉ dùng `New folder/master_dss_dataset.csv`.
- **input_1year/master_dss_dataset_1year.csv** – Redundant; master duy nhất là `New folder/master_dss_dataset.csv`.
- **FREE_GOLDAPI_HISTORY.csv**, **fetch_gold_from_free_apis.py** – Không dùng trong pipeline/training.
- **NEWS_SENTIMENT.csv** (root) – Nếu dùng sentiment thì đặt trong `input_1year/` và chạy script với output vào đó.
- **sjc_giavang_cache.csv**, **__pycache__/** – Cache/bytecode; bỏ qua bằng `.gitignore`.

## Chạy nhanh (từ root dự án)
1. Thu thập macro (tuỳ chọn): `python scripts/fetch_macro_1_year.py`
2. Chuẩn hoá input (tuỳ chọn): `python scripts/build_master_1year.py`
3. Build master: `python prepare_gold_dss_pipeline.py --output-file "New folder/master_dss_dataset.csv"` (mặc định đọc `input_1year/`).
4. Train: `cd "New folder" ; python train_xgboost_dss.py`
5. App: `python frontend/app.py` → http://localhost:5000
