import streamlit as st
import pandas as pd
import joblib
from src.data_prep import preprocess_and_align

st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Customer Churn Prediction System")
st.markdown("This application analyzes customer data to predict the risk of them leaving the bank.")
st.markdown("---")

st.sidebar.header("Enter Customer Information")

def user_input_features():
    credit_score = st.sidebar.slider("Credit Score", 350, 850, 650)
    age = st.sidebar.slider("Age", 18, 92, 35)
    tenure = st.sidebar.slider("Tenure", 0, 10, 5)
    balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 60000.0)
    num_of_products = st.sidebar.selectbox("Product Count", [1, 2, 3, 4])
    estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)
    
    geography = st.sidebar.selectbox("Country (Geography)", ["France", "Germany", "Spain"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    
    has_crcard = st.sidebar.radio("Credit Card Available", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    is_active = st.sidebar.radio("Active Member", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")

    data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_crcard,
        'IsActiveMember': is_active,
        'EstimatedSalary': estimated_salary,
        'RowNumber': 0, 'CustomerId': 0, 'Surname': 'Unknown'
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("Entered Customer Data")
st.write(input_df)

if st.button("🚀 Analyze Risk"):
    try:
        model = joblib.load('models/lgbm_final_model.pkl')
        expected_columns = joblib.load('models/model_columns.pkl')
        
        processed_data = preprocess_and_align(input_df, expected_columns)
        
        prediction_prob = model.predict_proba(processed_data)[:, 1][0]
        threshold = 0.60
        prediction = 1 if prediction_prob > threshold else 0
        
        st.markdown("---")
        st.subheader("Analysis Result")
        
        if prediction == 1:
            st.error(f"⚠️ WARNING: Customer is in a high risk group! (Churn)")
            st.write(f"Churn Probability: **%{prediction_prob*100:.2f}**")
            st.info("Suggestion: Special promotion offer should be made to this customer.")
        else:
            st.success(f"✅ SAFE: Customer is loyal.")
            st.write(f"Churn Probability: **%{prediction_prob*100:.2f}**")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please make sure the 'models' folder and files are complete.")