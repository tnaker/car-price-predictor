import pandas as pd

def preprocess_new_car(new_car: dict, feature_list: list) -> pd.DataFrame:
    df_new = pd.DataFrame([new_car])
    df_encoded = pd.get_dummies(df_new)

    # Thêm các cột thiếu
    missing_cols = list(set(feature_list) - set(df_encoded.columns))
    missing_df = pd.DataFrame(0, index=[0], columns=missing_cols)
    df_encoded = pd.concat([df_encoded, missing_df], axis=1)

    # Sắp xếp đúng thứ tự
    df_encoded = df_encoded[feature_list]
    return df_encoded