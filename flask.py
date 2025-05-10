from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load trained model
with open('credit.pkl', 'rb') as f:
    model = pickle.load(f)

# Risk level map
risk_map = {0: 'Low', 1: 'Medium', 2: 'High'}

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert input to DataFrame
    input_df = pd.DataFrame([data])

    # Select features
    features = ['Total_Deposits', 'Total_Withdrawals', 'Average_Balance', 'Transaction_Count', 'Credit_Score']
    X_input = input_df[features]

    # Predict risk level
    risk_pred = model.predict(X_input)[0]
    risk_label = risk_map[risk_pred]

    return jsonify({
        'Credit_Score': input_df['Credit_Score'].iloc[0],
        'Risk_Level': risk_label
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
