import streamlit as st
import pandas as pd
import joblib

MODEL_FILE = "decision_tree_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

def predict_proba(model, df):
    return model.predict_proba(df)[0][1]   # Probability of Delay (1)

st.set_page_config(page_title="E-Commerce Delivery Prediction", layout="wide")
st.title("📦 E-Commerce Product Delivery Predictor")

model = load_model()

st.sidebar.header("⚙ Prediction Control")
threshold = st.sidebar.slider("Decision Threshold", 0.10, 0.90, 0.50, 0.01)

col1, col2 = st.columns(2)

with col1:
    warehouse_block = st.selectbox("Warehouse Block", ['A', 'B', 'C', 'D', 'F'])
    mode = st.selectbox("Mode of Shipment", ['Ship', 'Flight', 'Road'])
    calls = st.slider("Customer Care Calls", 2, 7, 4)
    rating = st.select_slider("Customer Rating", [1, 2, 3, 4, 5], value=3)
    cost = st.number_input("Cost of Product (USD)", 50, 300, 150)

with col2:
    purchases = st.slider("Prior Purchases", 2, 10, 4)
    importance = st.radio("Product Importance", ['low', 'medium', 'high'], horizontal=True)
    gender = st.radio("Customer Gender", ['F', 'M'], horizontal=True)
    discount = st.number_input("Discount Offered (%)", 0, 65, 10)
    weight = st.number_input("Weight in Grams", 500, 8000, 2000)

if st.button("🔍 Predict"):
    input_data = pd.DataFrame({
        'Warehouse_block': [warehouse_block],
        'Mode_of_Shipment': [mode],
        'Customer_care_calls': [calls],
        'Customer_rating': [rating],
        'Cost_of_the_Product': [cost],
        'Prior_purchases': [purchases],
        'Product_importance': [importance],
        'Gender': [gender],
        'Discount_offered': [discount],
        'Weight_in_gms': [weight]
    })

    p_delay = predict_proba(model, input_data)
    result = 1 if p_delay >= threshold else 0

    if result == 0:
        st.success(f"✅ Product will *Reach On Time* (0)\n*Delay Probability:* {p_delay:.2f}")
        st.balloons()
    else:
        st.error(f"⚠ Product will *NOT Reach On Time* (1)\n*Delay Probability:* {p_delay:.2f}")

    st.info("Adjust the threshold in sidebar to change model decision sensitivity.")
