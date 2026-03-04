"""
Chạy test với model và scaler đã lưu: cùng pipeline data -> scale test -> predict -> in báo cáo.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from train_xgboost_dss import (
    INPUT_FILE,
    OUTPUT_DIR_NAME,
    add_technical_indicators,
    add_t3_target,
    chronological_split,
    engineer_features,
    load_and_resample_daily,
    split_xy,
)


def scale_test_with_saved_scaler(X_test: pd.DataFrame, scaler_path: Path) -> pd.DataFrame:
    """Transform X_test với scaler đã lưu (cùng cột scale như lúc train)."""
    scaler = joblib.load(scaler_path)
    one_hot = [c for c in X_test.columns if c.startswith("gold_code_")]
    scale_candidates = [c for c in X_test.columns if c not in one_hot]
    scale_columns = [c for c in scale_candidates if pd.api.types.is_numeric_dtype(X_test[c])]
    X_test = X_test.copy()
    if scale_columns:
        X_test.loc[:, scale_columns] = scaler.transform(X_test[scale_columns])
    return X_test


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / OUTPUT_DIR_NAME
    input_path = base_dir / INPUT_FILE
    model_path = output_dir / "xgboost_dss_model.pkl"
    scaler_path = output_dir / "scaler_xgboost_dss.pkl"

    if not model_path.exists() or not scaler_path.exists():
        print("Khong tim thay model hoac scaler. Chay train_xgboost_dss.py truoc.")
        return

    print("Load data va chay cung pipeline nhu luc train...")
    df = load_and_resample_daily(input_path)
    df = add_technical_indicators(df)
    label_encoder = LabelEncoder()
    label_encoder.fit(["BUY", "HOLD", "SELL"])
    df, _ = add_t3_target(df)
    df = engineer_features(df)

    train_df, test_df = chronological_split(df)
    X_train, X_test, y_train, y_test = split_xy(train_df, test_df)

    print("Load scaler va transform X_test...")
    X_test_scaled = scale_test_with_saved_scaler(X_test, scaler_path)

    print("Load model va predict...")
    model = joblib.load(model_path)
    y_pred = model.predict(X_test_scaled)

    print("\n" + "=" * 50)
    print("KET QUA TEST (test set)")
    print("=" * 50)
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))
    f1_macro = f1_score(y_test, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0)
    print(f"\nF1 macro:    {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")
    print(f"\nSo mau test: {len(y_test)}")


if __name__ == "__main__":
    main()
