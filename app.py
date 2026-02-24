import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib


DATA_PATH = 'diamonds.csv'
MODEL_PATH = 'best_model.joblib'
SCALER_PATH = 'scaler.joblib'
INFO_PATH = 'model_info.json'


def prepare_df(path=DATA_PATH):
    df = pd.read_csv(path)
    df.rename(columns={"x": "length", "y": "width", "z": "depth", "depth": "depth_percent", "table": "table_percent"}, inplace=True)
    df['L/W'] = df['length'] / df['width']
    df[['length', 'width', 'depth', 'L/W']] = df[['length', 'width', 'depth', 'L/W']].replace(0, np.NaN)
    df.dropna(inplace=True)
    return df


def ensure_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        # train model if not present
        try:
            subprocess.check_call([sys.executable, 'train_model.py'])
        except Exception as e:
            st.error('Failed to train model automatically. Run `python train_model.py` manually.')
            raise


st.sidebar.title('Diamond Price Prediction')
page = st.sidebar.selectbox('Choose page', ['Home', 'Predict'])

df = prepare_df()

# get feature ranges from dataset
min_carat, max_carat = float(df['carat'].min()), float(df['carat'].max())
min_depth, max_depth = float(df['depth'].min()), float(df['depth'].max())
min_length, max_length = float(df['length'].min()), float(df['length'].max())
min_width, max_width = float(df['width'].min()), float(df['width'].max())

if page == 'Home':
    st.title('Diamond Price Prediction')

    st.header('Project Overview')
    st.markdown(
        'This Streamlit app provides an interface to predict diamond prices (USD) using a trained regression model. '
        'Inputs are derived from the provided dataset and the model is loaded from the workspace when predictions are requested.'
    )

    st.subheader('Business objective')
    st.write(
        '- **Goal:** Provide fast, interpretable price estimates for diamonds given basic physical attributes so stakeholders '
        'can make pricing or valuation decisions.'
        '- **Primary users:** Retail jewelers, appraisers, and analysts.'
    )

    st.subheader('Key project features')
    st.write('- Predict price from physical measurements: `carat`, `depth`, `length`, `width`.')
    st.write('- Uses a trained regression model with a scaler to normalize inputs.')
    st.write('- Displays model performance (MAE) from training metadata when available.')
    st.write('- Simple UI for quick what-if scenarios via sliders.')

    st.header('Data overview')
    st.write(f'- Rows: {df.shape[0]}')
    st.write(f'- Columns: {df.shape[1]}')
    st.write('- Features used for prediction: `carat`, `depth`, `length`, `width`')

    st.subheader('Feature ranges (from dataset)')
    st.write(f'- Carat: {min_carat} — {max_carat}')
    st.write(f'- Depth: {min_depth} — {max_depth}')
    st.write(f'- Length: {min_length} — {max_length}')
    st.write(f'- Width: {min_width} — {max_width}')

    st.subheader('Summary statistics')
    st.dataframe(df.describe())

    st.subheader('Price distribution')
    try:
        fig, ax = plt.subplots()
        ax.hist(df['price'], bins=30, color='#2c7fb8')
        ax.set_xlabel('Price (USD)')
        ax.set_ylabel('Count')
        ax.set_title('Price distribution')
        st.pyplot(fig)
    except Exception:
        st.write('Unable to render price distribution plot.')

    st.subheader('Preview')
    st.dataframe(df.head())

    if os.path.exists(INFO_PATH):
        try:
            with open(INFO_PATH, 'r') as f:
                info = json.load(f)
            mae = info.get('mae')
            if mae is not None:
                st.info(f'Model MAE (from training): {mae:.6f}')
        except Exception:
            pass

elif page == 'Predict':
    st.title('Predict')
    st.write('Enter diamond features; model runs in backend and predicts price (USD).')

    st.sidebar.header('Input features')
    carat = st.sidebar.slider('Carat', min_value=min_carat, max_value=max_carat, value=float(df['carat'].median()))
    depth = st.sidebar.slider('Depth (mm)', min_value=min_depth, max_value=max_depth, value=float(df['depth'].median()))
    length = st.sidebar.slider('Length (mm)', min_value=min_length, max_value=max_length, value=float(df['length'].median()))
    width = st.sidebar.slider('Width (mm)', min_value=min_width, max_value=max_width, value=float(df['width'].median()))

    if st.button('Predict'):
        ensure_model()
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        X = np.array([[carat, depth, length, width]])
        X_scaled = scaler.transform(X)
        pred_log = model.predict(X_scaled)
        try:
            price = float(np.exp(pred_log)[0])
        except Exception:
            price = float(np.exp(pred_log))

        st.success(f'Predicted price: ${price:,.2f}')

        # show MAE from training info if available (no algorithm name shown)
        if os.path.exists(INFO_PATH):
            try:
                with open(INFO_PATH, 'r') as f:
                    info = json.load(f)
                mae = info.get('mae')
                if mae is not None:
                    st.write(f'Model MAE (from training): {mae:.6f}')
            except Exception:
                pass
