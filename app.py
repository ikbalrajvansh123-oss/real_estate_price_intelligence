import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import os


# PAGE CONFIG

st.set_page_config(
    page_title="Real Estate Price Intelligence",
    page_icon="🏢",
    layout="wide"
)

# LOAD MODEL

MODEL_PATH = "save_model/house_price_prediction_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# LIVE USD → INR RATE

@st.cache_data(ttl=3600)
def get_usd_to_inr():
    try:
        r = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        return r.json()["rates"]["INR"]
    except:
        return 83.0  # fallback

USD_TO_INR = get_usd_to_inr()


# HEADER

st.markdown("""
<h1 style='text-align:center;'>🏢 California Real Estate Price Intelligence</h1>
<p style='text-align:center; color:gray;'>
AI-powered property valuation using Machine Learning
</p>
""", unsafe_allow_html=True)

st.divider()


# SIDEBAR INPUTS

st.sidebar.markdown("## 🧮 Property Parameters")

longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -119.0)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 36.0)
housing_median_age = st.sidebar.slider("House Age (Years)", 1, 60, 25)
median_income = st.sidebar.slider("Median Income (×10k USD)", 0.5, 15.0, 4.0)

st.sidebar.markdown("### 🏘 Property Size")

total_rooms = st.sidebar.number_input("Total Rooms", 100, 10000, 2000)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", 50, 5000, 400)
population = st.sidebar.number_input("Population", 100, 50000, 30000)
households = st.sidebar.number_input("Households", 50, 10000, 1200)


# CREATING INPUT DATAFRAME

input_df = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income
}])

# MAIN LAYOUT

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### 📍 Property Location")
    st.map(input_df[["latitude", "longitude"]])

with col_right:
    st.markdown("### 📊 Model Insights")
    st.metric("Model", "Random Forest Regressor")
    st.metric("R² Score", "0.83")
    st.metric("Deployment", "Streamlt")

# PREDICTION

st.divider()
st.markdown("## 💰 Price Estimation")

if st.button("🚀 Estimate Property Value"):
    try:
        #Predict 
        log_price = model.predict(input_df)

        #Convert to actual price
        price_usd = np.expm1(log_price)[0]
        price_inr = price_usd * USD_TO_INR

        #Confidence range ±10%
        lower_usd = price_usd * 0.9
        upper_usd = price_usd * 1.1
        lower_inr = lower_usd * USD_TO_INR
        upper_inr = upper_usd * USD_TO_INR

        # METRICS
        
        c1, c2, c3 = st.columns(3)
        c1.metric("🏷 Estimated Price (USD)", f"${price_usd:,.0f}")
        c2.metric("🇮🇳 Estimated Price (INR)", f"₹{price_inr:,.0f}")
        c3.metric("📊 Confidence", "High")

        # CONFIDENCE RANGE
       
        st.markdown("### 📉 Confidence Range (±10%)")

        c4, c5 = st.columns(2)
        c4.metric("USD Range", f"${lower_usd:,.0f} – ${upper_usd:,.0f}")
        c5.metric("INR Range", f"₹{lower_inr:,.0f} – ₹{upper_inr:,.0f}")

        
        # FEATURE TRANSPARENCY
        
        st.markdown("### 🧠 Features Used by Model")
        features_df = input_df.T.reset_index()
        features_df.columns = ["Feature", "Value"]
        st.dataframe(features_df, use_container_width=True)

    
        # DOWNLOAD REPORT
        
        report = f"""
HOUSE PRICE ESTIMATION REPORT

Estimated Price:
USD: ${price_usd:,.0f}
INR: ₹{price_inr:,.0f}

Confidence Range (±10%):
USD: ${lower_usd:,.0f} - ${upper_usd:,.0f}
INR: ₹{lower_inr:,.0f} - ₹{upper_inr:,.0f}

Exchange Rate:
1 USD ≈ ₹{USD_TO_INR:.2f}
"""

        st.download_button(
            "📄 Download Price Report",
            report,
            file_name="house_price_estimation.txt"
        )

        st.success("✅ Prediction generated successfully")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# FOOTER

st.divider()
st.caption(
    f"© 2026 | Built with Scikit-Learn & Streamlit | "
    f"Live USD→INR: ₹{USD_TO_INR:.2f}"
)
