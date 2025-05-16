import streamlit as st
import joblib
import numpy as np

# --- Load your trained components ---
model = joblib.load("rf_model.joblib")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoders.pkl")

# --- App Title ---
st.title("🕒 Food Delivery Duration Estimator")
st.markdown("Enter order details below to estimate delivery time in **minutes**.")

# --- Input Fields ---
market_id = st.selectbox("Market ID", [1, 2, 3, 4, 5, 6, 7])  # add more if needed
store_category = st.selectbox("Store Primary Category", label_encoder.classes_)
order_protocol = st.selectbox("Order Protocol", [0, 1, 2, 3, 4])
total_items = st.number_input("Total Items", min_value=1, value=3)
subtotal = st.number_input("Subtotal (in cents)", min_value=100, value=1500)
num_distinct_items = st.number_input("Number of Distinct Items", min_value=1, value=2)
min_item_price = st.number_input("Minimum Item Price", min_value=1, value=500)
max_item_price = st.number_input("Maximum Item Price", min_value=1, value=1200)
onshift = st.number_input("Total Onshift Partners", min_value=0, value=5)
busy = st.number_input("Total Busy Partners", min_value=0, value=2)
outstanding = st.number_input("Total Outstanding Orders", min_value=0, value=3)

# --- Prediction Button ---
if st.button("Predict Delivery Duration"):
    try:
        # Encode & scale inputs
        store_encoded = label_encoder.transform([store_category])[0]
        input_data = np.array([[market_id, store_encoded, order_protocol, total_items, subtotal,
                                num_distinct_items, min_item_price, max_item_price,
                                onshift, busy, outstanding]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        st.success(f"📦 Estimated Delivery Duration: **{prediction:.2f} minutes**")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")
