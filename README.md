# Diamond Price Prediction - Streamlit App

This project provides a Streamlit app to predict diamond prices using features from `diamonds.csv`.

Quick steps

1. Create a Python environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model (this trains multiple algorithms and selects the best by lowest MAE):

```bash
python train_model.py
```

This will create `best_model.joblib`, `scaler.joblib`, and `model_info.json` in the project folder.

4. Run the Streamlit app:

```bash
streamlit run app.py
```

Notes

- The app reads min/max values for the input features from `diamonds.csv` and uses those ranges for the input sliders.
- The app selects and loads the trained model automatically; the algorithm itself is kept in the backend and not displayed in the UI.
- The training script uses the same preprocessing and feature selection as the provided notebook (`carat`, `depth`, `length`, `width`) and trains Linear Regression and Random Forest, selecting the one with the lower MAE.
