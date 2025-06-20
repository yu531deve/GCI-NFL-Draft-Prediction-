# src/features.py

def add_custom_bmi(df):
    """
    身長と体重からBMI的な指標を計算し、CustomBMI列として追加する。
    """
    height_m = df["Height"] / 100  # cm → m
    df["CustomBMI"] = df["Weight"] / (height_m ** 2)
    return df
