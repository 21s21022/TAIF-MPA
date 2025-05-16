import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🕒 Delivery Duration Predictor")

st.markdown("Enter the following order details:")

# Input fields
store_category = st.selectbox("Store Primary Category", le.classes_)
order_protocol = st.selectbox("Order Protocol", [0, 1, 2, 3, 4])
total_items = st.number_input("Total Items", min_value=1, max_value=100, value=3)
subtotal = st.number_input("Subtotal (in cents)", min_value=100, value=1500)
num_distinct_items = st.number_input("Number of Distinct Items", min_value=1, value=2)
min_item_price = st.number_input("Minimum Item Price", min_value=1, value=500)
max_item_price = st.number_input("Maximum Item Price", min_value=1, value=1200)
onshift = st.number_input("Total Onshift Partners", min_value=0, value=5)
busy = st.number_input("Total Busy Partners", min_value=0, value=2)
outstanding = st.number_input("Total Outstanding Orders", min_value=0, value=3)

if st.button("Predict Delivery Duration"):
    # Prepare input data
    cat_encoded = le.transform([store_category])[0]
    input_data = np.array([[cat_encoded, order_protocol, total_items, subtotal,
                            num_distinct_items, min_item_price, max_item_price,
                            onshift, busy, outstanding]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"Estimated Delivery Duration: **{prediction:.2f} minutes**")
