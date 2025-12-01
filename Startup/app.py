# app.py
import streamlit as st
import pickle
import pandas as pd
import base64

# ==========================
# Background Image + Text Style
# ==========================
def add_bg_image(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* Make all text white for readability */
        .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp label {{
            color: white !important;
        }}
        /* Input text black */
        .stTextInput>div>div>input {{
            color: black !important;
        }}
        .stNumberInput>div>div>input {{
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_image("startup.jpg")

# ==========================
# Load model and encoders
# ==========================
@st.cache_resource
def load_model():
    with open("Startup_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("industry_encoder.pkl", "rb") as f:
        industry_le = pickle.load(f)
    with open("region_encoder.pkl", "rb") as f:
        region_le = pickle.load(f)
    return model, industry_le, region_le

model, industry_le, region_le = load_model()

# ==========================
# App Title
# ==========================
st.title("Startup Profitability Predictor")
st.markdown("Enter startup details to predict if the startup is profitable.")

# ==========================
# Single Startup Prediction Form
# ==========================
with st.form("startup_form"):
    industry = st.selectbox("Industry", industry_le.classes_)
    region = st.selectbox("Region", region_le.classes_)
    funding_rounds = st.number_input("Funding Rounds", min_value=0, value=1)
    funding_amount = st.number_input("Funding Amount (M USD)", min_value=0.0, value=0.0)
    valuation = st.number_input("Valuation (M USD)", min_value=0.0, value=0.0)
    revenue = st.number_input("Revenue (M USD)", min_value=0.0, value=0.0)
    employees = st.number_input("Number of Employees", min_value=1, value=1)
    market_share = st.number_input("Market Share (%)", min_value=0.0, max_value=100.0, value=0.0)
    year_founded = st.number_input("Year Founded", min_value=1900, max_value=2025, value=2020)

    submitted = st.form_submit_button("Predict")

# ==========================
# Single Prediction
# ==========================
if submitted:
    try:
        # Encode categorical inputs
        industry_val = industry_le.transform([industry])[0]
        region_val = region_le.transform([region])[0]

        # Prepare DataFrame matching model features and order
        feature_cols = ['Industry', 'Funding Rounds', 'Funding Amount (M USD)',
                        'Valuation (M USD)', 'Revenue (M USD)', 'Employees',
                        'Market Share (%)', 'Year Founded', 'Region']

        input_df = pd.DataFrame([[
            industry_val,
            funding_rounds,
            funding_amount,
            valuation,
            revenue,
            employees,
            market_share,
            year_founded,
            region_val
        ]], columns=feature_cols)

        # Predict
        pred = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0][1]

        result = "✅ Profitable" if pred == 1 else "❌ Not Profitable"
        st.subheader(f"Prediction: {result}")
        st.write(f"Probability of being profitable: {pred_proba*100:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")
