# Hướng Khắc Phục Lỗi Kết Nối Server

## Vấn Đề
Không thể kết nối đến Flask server http://127.0.0.1:5000

## Nguyên Nhân Có Thể
1. **Sklearn version mismatch** - Model được train với sklearn 1.6.1 nhưng bạn cài sklearn 1.8.0
2. **Port 5000 bị chiếm** - Có process khác đang dùng port 5000
3. **Dependencies không đầy đủ** - Thiếu package hoặc version sai
4. **Firewall** - Windows Firewall chặn kết nối

## Giải Pháp

### 1. **CÀI ĐẶT DEPENDENCIES TỪ requirements.txt**
```bash
# Từ thư mục gốc:
cd d:\EXE112\anhthuongemnhatma
pip install -r requirements.txt
```

### 2. **KIỂM TRA VÀ GIẢI PHÓNG PORT 5000**
```powershell
# Kiểm tra process đang dùng port 5000
netstat -ano | findstr ":5000"

# Nếu có process, kill nó (thay xxxxx bằng PID):
taskkill /PID xxxxx /F

# Hoặc đơn giản: đóng tất cả cửa sổ Terminal/Python đủ lâu để kernel release port
```

### 3. **CHẠY APP**
```bash
python frontend/app.py
```

Output sẽ hiển thị:
```
Open http://127.0.0.1:5000 in browser
 * Running on http://127.0.0.1:5000
```

### 4. **KIỂM TRA APP CHẠY**
Mở **Command Prompt** hoặc **PowerShell** thứ 2 (KHÔNG kập app.py đang chạy), rồi:
```bash
# Test API endpoints
python -c "import urllib.request, json; r = urllib.request.urlopen('http://127.0.0.1:5000/api/gold_codes'); print(json.loads(r.read()))"
```

Hoặc mở browser và truy cập:
- **Homepage**: http://127.0.0.1:5000/
- **API Gold Codes**: http://127.0.0.1:5000/api/gold_codes
- **API Macro**: http://127.0.0.1:5000/api/macro
- **API Predict**: http://127.0.0.1:5000/api/predict

## Lưu Ý Quan Trọng

### ⚠️ Sklearn Warning (An toàn để bỏ qua)
Models được train với sklearn 1.6.1 nhưng nếu bạn cài 1.8.0, sẽ thấy:
```
InconsistentVersionWarning: Trying to unpickle estimator LabelEncoder from version 1.6.1 when using version 1.8.0
```
Đây là **cảnh báo chứ không phải lỗi**. Để tránh, cài requirements.txt với sklearn 1.6.1:
```bash
pip install scikit-learn==1.6.1
```

### 📂 Cấu Trúc File Cần Thiết
```
d:\EXE112\anhthuongemnhatma\
├── frontend\
│   ├── app.py                    ✓
│   └── templates\
│       └── index.html            ✓
├── New folder\
│   ├── output\
│   │   ├── xgboost_dss_model.pkl ✓
│   │   ├── scaler_xgboost_dss.pkl ✓
│   │   └── label_encoder_dss.pkl ✓
│   ├── train_xgboost_dss.py      ✓
│   └── preprocessing.py          ✓
├── master_dss_dataset.csv        ✓
├── MACRO_FEATURES_1_YEAR.csv     ✓
└── requirements.txt              ✓ (Vừa tạo)
```

## Nếu Vẫn Không Hoạt động
1. **Kiểm tra Python version**: `python --version` (nên ≥3.9)
2. **Kiểm tra file tồn tại**: Đảm bảo tất cả file .csv và .pkl tồn tại
3. **Xem logs**: Chạy `python frontend/app.py` và kiểm tra error message cụ thể
4. **Thử port khác** (tạm thời test): Sửa file `app.py` dòng cuối: `app.run(host="0.0.0.0", port=5001, debug=False)`

---
Đã cài đặt **requirements.txt** cho bạn. Hãy chạy lệnh trên để fix! 🚀
