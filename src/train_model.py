import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(inplace=True)
    df.drop(['Make', 'Number of Doors', 'Market Category'], axis=1, inplace=True)
    category_cols = df.select_dtypes(include=['object']).columns.tolist()

    df_encoded = pd.get_dummies(df, columns=category_cols, drop_first=True)
    df_encoded = df_encoded.astype({col: 'int' for col in df_encoded.select_dtypes(include='bool').columns})
    return df_encoded

def train():
    # Load raw data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'car_price_data_origin.csv')

    df = pd.read_csv(data_path)
    df_encoded = preprocess_dataframe(df)

    # Tách X và y
    X = df_encoded.drop('MSRP', axis=1)
    y = df_encoded['MSRP']

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Đánh giá
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(" RMSE:", round(rmse, 2))

    # Lưu model & features
    model_dir = os.path.join(base_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'car_price_model.pkl'))
    joblib.dump(X_train.columns.tolist(), os.path.join(model_dir, 'features.pkl'))

if __name__ == '__main__':
    train()