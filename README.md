# Customer Churn Prediction

End-to-end workflow to predict bank customer churn on a 10k-row dataset. Includes EDA, preprocessing, feature engineering, model benchmarking, hyperparameter tuning (Optuna + LightGBM), and a runnable pipeline for batch predictions. Built on [Churn for Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers/data) dataset

[Try The Application](https://mgfinalproject-l4j2rwlfypnf5mmgknennu.streamlit.app/)

## Project Structure

```
CustomerChurnPredictionSystem/
├── app.py              
├── main_pipeline.py                      
├── requirements.txt      
├── Readme.md            
├── data/
│  ├── processed/
│  └── raw/ 
├── models/ 
└──notebooks/
│  ├── 01_eda.ipynb              
│  ├── 02_preprocessing_feature_eng.ipynb                  
│  ├── 03_model_benchmarking.ipynb
│  └── 04_hyperparameter_tuning.ipynb
├── reports/
└──src/   
   ├── data_prep.py             
   ├── features.py                
   └── visual.py
```

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

