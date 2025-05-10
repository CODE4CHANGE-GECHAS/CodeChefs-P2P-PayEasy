import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load modified dataset
df = pd.read_csv('modified_credit_scoring_dataset_ready.csv')

# Encode risk level (target)
df['Risk_Level_Code'] = df['Risk_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Features and target
X = df[['Total_Deposits', 'Total_Withdrawals', 'Average_Balance', 'Transaction_Count', 'Credit_Score']]
y = df['Risk_Level_Code']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open('credit_risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'credit_risk_model.pkl'")
