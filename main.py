import streamlit as st
import numpy as np
import joblib

# Load model and artifacts
model = joblib.load("xgb_fraud_model.pkl")
le = joblib.load("type_label_encoder.pkl")
selected_features = joblib.load("model_features.pkl")

st.title("Fraud Detection Streamlit App")

st.write("Please enter all transaction details below:")

# Ask user for all required details
step = st.number_input(
    "Step (1 step = 1 hour, range: 1-744)", min_value=1, max_value=744, value=1
)
amount_to_oldbalance_orig_ratio = st.number_input(
    "Amount to Old Balance Orig Ratio (amount / oldbalanceOrg)", min_value=0.0, value=1.0
)
balance_change_dest = st.number_input(
    "Balance Change Dest (newbalanceDest - oldbalanceDest)", value=0.0
)
balance_change_orig = st.number_input(
    "Balance Change Orig (oldbalanceOrg - newbalanceOrig)", value=0.0
)
transaction_pattern_encoded = st.selectbox(
    "Transaction Pattern Encoded",
    [0, 1, 2, 3],
    format_func=lambda x: [
        "Customer to Customer (0)",
        "Customer to Merchant (1)",
        "Merchant to Customer (2)",
        "Merchant to Merchant (3)"
    ][x]
)
type_input = st.selectbox(
    "Type of Transaction",
    ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT", "CASH_IN"]
)
amount = st.number_input(
    "Transaction Amount", min_value=0.0, value=100.0
)
dest_is_merchant = st.selectbox(
    "Is Destination a Merchant?",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# Encode 'type'
type_encoded = le.transform([type_input])[0]

# Prepare input in the correct order
input_data = np.array([[step,
                        amount_to_oldbalance_orig_ratio,
                        balance_change_dest,
                        balance_change_orig,
                        transaction_pattern_encoded,
                        type_encoded,
                        amount,
                        dest_is_merchant]])

if st.button("Predict Fraud"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ Warining this transaction is FRAUD.")
    else:
        st.success("✅ Congrats this transaction is Success.")