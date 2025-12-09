# Customer Churn Prediction

End-to-end workflow to predict bank customer churn on a 10k-row dataset. Includes EDA, preprocessing, feature engineering, model benchmarking, hyperparameter tuning (Optuna + LightGBM), and a runnable pipeline for batch predictions. Built on [Churn for Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers/data) dataset

[Try The Application](https://mgfinalproject-l4j2rwlfypnf5mmgknennu.streamlit.app/)

## Project Structure
- `data/raw/main_data.csv` — original dataset
- `data/processed/main_data_processed.csv` — processed versions used in notebooks
- `notebooks/`
  - `01_eda.ipynb` — dataset overview, target imbalance (~80/20), categorical churn visuals
  - `02_preprocessing_feature_eng.ipynb` — encoding, scaling, feature engineering, save processed CSV
  - `03_model_benchmarking.ipynb` — multiple models (class weights), F1/ROC-AUC comparison, threshold tuning
  - `04_hyperparameter_tuning.ipynb` — Optuna + LightGBM search, best params, threshold tuning, test eval
- `src/`
  - `data_prep.py` — feature_engineering_pipeline, preprocess_and_align (OHE + column alignment)
  - `visual.py` — churn plots, outlier helpers
  - `features.py` — (placeholder)
- `models/` — trained LightGBM model (`lgbm_final_model.pkl`) and expected columns
- `reports/` — sample churn prediction report CSV
- `main_pipeline.py` — batch prediction pipeline using saved model
- `requirements.txt` — dependencies

## Notebooks
1) **EDA (`01_eda.ipynb`)**
   - Explore schema, imbalance, categorical churn distributions.
2) **Preprocessing & Feature Engineering (`02_preprocessing_feature_eng.ipynb`)**
   - One-hot encode categorical vars, create engineered features, save processed CSV to `data/processed/`.
3) **Benchmarking (`03_model_benchmarking.ipynb`)**
   - Models with class weights (LogReg, Tree, RF, XGBoost, LightGBM). Compare via F1/ROC-AUC. Threshold tuning for better recall/precision balance.
4) **Hyperparameter Tuning (`04_hyperparameter_tuning.ipynb`)**
   - Optuna search for LightGBM (F1 objective via CV). Train final model, tune decision threshold, evaluate on test. Saves model + column list to `models/`.

## Key Metrics (sample)
- Best tuned LightGBM (CV F1 ≈ 0.619)
- Threshold tuning lifted train F1 to ~0.745
- Test F1 ~0.63 at threshold 0.60 with balanced precision/recall trade-off.

