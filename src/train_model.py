import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(inplace=True)
    df.drop(['Model', 'Number of Doors'], axis=1, inplace=True)
    category_cols = df.select_dtypes(include=['object']).columns.tolist()

    df_encoded = pd.get_dummies(df, columns=category_cols, drop_first=True)
    df_encoded = df_encoded.astype({col: 'int' for col in df_encoded.select_dtypes(include='bool').columns})
    return df_encoded

def train():
    # Load raw data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'car_price_data_origin.csv')

    print("--- Đang đọc dữ liệu ---")
    df = pd.read_csv(data_path)
    df_encoded = preprocess_dataframe(df)

    # Tách X và y
    X = df_encoded.drop('MSRP', axis=1)
    y = np.log1p(df_encoded['MSRP'])

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    print("--- Đang huấn luyện (Random Forest + Log Transform) ---")
    model = RandomForestRegressor(random_state=25, n_jobs=-1)
    model.fit(X_train, y_train)

    # Đánh giá
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))
    mape = np.mean(np.abs((np.expm1(y_test)- np.expm1(y_pred)) / np.expm1(y_test))) * 100
    # --- KIỂM TRA RIÊNG NHÓM XE BÌNH DÂN (Dưới 30k USD) ---
    # Tạo bảng so sánh tạm thời
    comparison_df = pd.DataFrame({'Actual': np.expm1(y_test), 'Predicted':np.expm1(y_pred)})
    
    # Lọc ra những xe giá rẻ (dưới 30,000 USD)
    cheap_cars = comparison_df[comparison_df['Actual'] < 30000]
    
    # Tính sai số trung bình của nhóm này
    cheap_rmse = np.sqrt(mean_squared_error(cheap_cars['Actual'], cheap_cars['Predicted']))
    
    print(f"\n🚙 Với xe bình dân (<30k USD):")
    print(f"   Sai số trung bình chỉ là: {int(cheap_rmse)} USD")
    print("RMSE:", round(rmse, 2))
    print(f"📊 Sai số phần trăm trung bình (MAPE): {round(mape, 2)}%")

    # Lưu model & features
    model_dir = os.path.join(base_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'car_price_model.pkl'))
    joblib.dump(X_train.columns.tolist(), os.path.join(model_dir, 'features.pkl'))

if __name__ == '__main__':
    train()