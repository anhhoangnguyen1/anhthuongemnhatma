# Hướng thay đổi nhãn và model (đã áp dụng)

## 1. Logic nhãn (`add_t3_target`)

### Chế độ mặc định (không dùng `--hold-band`)
- **BUY:**  `expected_profit > buy_ratio * spread` (mặc định buy_ratio=1.0)
- **SELL:** `expected_profit < -sell_ratio * spread` (mặc định sell_ratio=0.3)
- **HOLD:** Phần còn lại (vùng giữa rộng → dễ bị model bỏ qua)

### Chế độ vùng HOLD đối xứng (`--hold-band R`)
- **HOLD:** `|expected_profit| <= R * spread` (chỉ “biến động gần 0”)
- **BUY:**  `expected_profit > R * spread`
- **SELL:** `expected_profit < -R * spread`

Giúp cân bằng 3 lớp hơn, HOLD rõ nghĩa (không vào/ra vị thế mạnh).

**Ví dụ:** `python train_xgboost_dss.py --hold-band 0.2`  
→ HOLD khi |lợi nhuận 3 ngày| ≤ 20% spread; BUY/SELL khi vượt ±20% spread.

---

## 2. Tham số dòng lệnh

| Tham số        | Mặc định | Ý nghĩa |
|----------------|----------|--------|
| `--sell-ratio` | 0.3      | SELL khi lỗ > 30% spread (bỏ qua nếu dùng `--hold-band`) |
| `--buy-ratio`  | 1.0      | BUY khi lãi > 100% spread (bỏ qua nếu dùng `--hold-band`) |
| `--hold-band`  | None     | Bật vùng HOLD đối xứng (vd 0.2) |

---

## 3. Thay đổi hyperparameter và trọng số

- **Regularization:** `reg_alpha=1.5`, `reg_lambda=10` (tăng so với trước) → hạn chế overfit.
- **Cây đơn giản hơn:** `max_depth` chỉ [2, 3], `min_child_weight` [5, 10].
- **Trọng số HOLD:** Trong `tune_and_train_xgboost`, lớp HOLD (index 1) được nhân thêm `hold_weight_extra=1.3` để model ưu tiên đoán đúng HOLD, giảm tình trạng “gần như không bao giờ đoán HOLD”.

---

## 4. Gợi ý thứ tự thử

1. **Giữ mặc định (chỉ dùng thay đổi regularization + HOLD weight):**
   ```bash
   python train_xgboost_dss.py
   ```

2. **Thử vùng HOLD đối xứng (cân bằng 3 lớp):**
   ```bash
   python train_xgboost_dss.py --hold-band 0.2
   ```
   Hoặc `--hold-band 0.25` nếu muốn vùng HOLD rộng hơn.

3. **Kiểm tra phân bố nhãn trước khi train:**
   ```bash
   python train_xgboost_dss.py --check-labels
   ```

4. **Đánh giá nhiều cách chia:**
   ```bash
   python train_xgboost_dss.py --multi-split-eval
   ```

---

## 5. Nếu vẫn kém

- Thu thập thêm dữ liệu (nhiều tháng, nhiều mã vàng).
- Giảm số feature (chọn theo feature importance sau mỗi lần train).
- Thử `hold_weight_extra` lớn hơn (vd 1.5) trong code nếu vẫn thiên về BUY/SELL.
