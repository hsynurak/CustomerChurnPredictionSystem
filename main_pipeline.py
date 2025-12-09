import pandas as pd
import joblib
import os
from src.data_prep import preprocess_and_align

class ChurnPredictionPipeline:
     def __init__(self, model_path='models/lgbm_final_model.pkl', columns_path='models/model_columns.pkl'):
          if not os.path.exists(model_path) or not os.path.exists(columns_path):
               raise FileNotFoundError("Model or columns file not found")
               
          print("Model and requirements loaded successfully.")
          self.model = joblib.load(model_path)
          self.expected_columns = joblib.load(columns_path)
          self.threshold = 0.60  
          print("System ready.")
     
     def run(self, data_path):
          print(f"{data_path} reading...")
          try:
               raw_data = pd.read_csv(data_path)
          except Exception as e:
               return f"Warning: File not read. {str(e)}"

          customer_ids = raw_data['CustomerId'] if 'CustomerId' in raw_data.columns else raw_data.index

          print("Data processing and alignment...")
          processed_data = preprocess_and_align(raw_data, self.expected_columns)

          print("Prediction...")
          probs = self.model.predict_proba(processed_data)[:, 1]
          predictions = (probs > self.threshold).astype(int)

          results = pd.DataFrame({
            'CustomerId': customer_ids,
            'Churn_Probability': probs.round(4),
            'Churn_Prediction': predictions
          })

          risk_report = results[results['Churn_Prediction'] == 1].sort_values(by='Churn_Probability', ascending=False)

          return results, risk_report

if __name__ == "__main__":

     pipeline = ChurnPredictionPipeline()

     input_file = 'data/raw/main_data.csv' 
          
     all_results, risky_customers = pipeline.run(input_file)
          
     output_path = 'reports/churn_prediction_report.csv'
     all_results.to_csv(output_path, index=False)
          
     print("-" * 50)
     print(f"Pipeline completed successfully!")
     print(f"Total customers: {len(all_results)}")
     print(f"Number of risky customers: {len(risky_customers)}")
     print(f"Report saved to: {output_path}")
     print("-" * 50)
     print("First 5 risky customers:")
     print(risky_customers.head())