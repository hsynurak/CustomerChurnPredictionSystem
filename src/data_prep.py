import pandas as pd
import numpy as np

def feature_engineering_pipeline(df):
     
     df = df.copy()

     geo_mean_map = {
        'Germany': 119730.1, 
        'France': 62092.6, 
        'Spain': 61818.1
     }

     df["TenureOverAge"] = df["Tenure"] / df["Age"]
     df["BalanceOverSalary"] = df["Balance"] / df["EstimatedSalary"]
     df["Credit_Age_Ratio"] = df["CreditScore"] / df["Age"]
     df["Balance_Per_Product"] = df["Balance"] / df["NumOfProducts"]
     df["Active_Monetary_Value"] = df["Balance"] * df["IsActiveMember"]
     df["Has_Balance"] = df["Balance"].apply(lambda x: 1 if x > 0 else 0)
     global_mean = 76485.8
     df["Geo_Balance_Mean"] = df["Geography"].map(geo_mean_map).fillna(global_mean)     
     df['Balance_Rel_Geo'] = df['Balance'] / df['Geo_Balance_Mean']
     df["Age_Group"] = pd.cut(
          df["Age"], 
          bins=[18, 30, 45, 60, 100], 
          labels=["0", "1", "2", "3"], 
          include_lowest=True)
     df["Credit_Score_Tier"] = pd.cut(
          df["CreditScore"], 
          bins=[350, 550, 650, 750, 850], 
          labels=["0", "1", "2", "3"], 
          include_lowest=True)

     df["Age_Group"] = df["Age_Group"].astype(float)
     df["Credit_Score_Tier"] = df["Credit_Score_Tier"].astype(float)

     df = df.drop(["CustomerId","Surname","RowNumber", "HasCrCard", "Tenure"], axis = 1)

     return df

def preprocess_and_align(df, expected_columns):
     df = feature_engineering_pipeline(df)
     df = pd.get_dummies(df, columns=['Geography', 'Gender', 'IsActiveMember'], drop_first=True)
     df_final = df.reindex(columns=expected_columns, fill_value=0)
     return df_final