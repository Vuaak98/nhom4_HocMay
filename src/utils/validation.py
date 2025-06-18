import pandas as pd

def validate_dataframe(df: pd.DataFrame, required_columns: list, df_name: str):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame '{df_name}' bị thiếu các cột bắt buộc: {missing_cols}")
    print(f"✅ DataFrame '{df_name}' đã được xác thực thành công.") 