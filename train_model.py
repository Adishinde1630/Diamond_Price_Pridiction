import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib


def load_and_prepare(path="diamonds.csv"):
    df = pd.read_csv(path)
    # rename columns to match notebook
    df.rename(columns={"x": "length", "y": "width", "z": "depth", "depth": "depth_percent", "table": "table_percent"}, inplace=True)
    # create L/W and replace zeros with NaN then drop
    df['L/W'] = df['length'] / df['width']
    df[['length', 'width', 'depth', 'L/W']] = df[['length', 'width', 'depth', 'L/W']].replace(0, np.NaN)
    df.dropna(inplace=True)
    # target transform used in notebook
    df['Price_log'] = np.log(df['price'])
    X = df[['carat', 'depth', 'length', 'width']].copy()
    y = df['Price_log'].copy()
    return X, y


def train_and_select(X, y, out_dir="."):
    cols = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=cols, index=X_test.index)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    mae_lr = metrics.mean_absolute_error(y_test, y_pred_lr)

    # Random Forest
    rf = RandomForestRegressor(random_state=0)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    mae_rf = metrics.mean_absolute_error(y_test, y_pred_rf)

    # choose best (lowest MAE)
    if mae_lr <= mae_rf:
        best_model = lr
        best_name = "LinearRegression"
        best_mae = float(mae_lr)
    else:
        best_model = rf
        best_name = "RandomForestRegressor"
        best_mae = float(mae_rf)

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'best_model.joblib')
    scaler_path = os.path.join(out_dir, 'scaler.joblib')
    info_path = os.path.join(out_dir, 'model_info.json')

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    info = {
        "best_model": best_name,
        "mae": best_mae
    }
    with open(info_path, 'w') as f:
        json.dump(info, f)

    print(f"Saved best model: {best_name} with MAE: {best_mae:.6f}")
    print(f"Model files: {model_path}, {scaler_path}, {info_path}")


if __name__ == '__main__':
    X, y = load_and_prepare('diamonds.csv')
    train_and_select(X, y, out_dir='.')
