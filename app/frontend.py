import streamlit as st
import requests
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Ames Housing AI",
    page_icon="🏠",
    layout="wide"
)

# Title and Header
st.title("🏠 Ames House Price Predictor & Planner")
st.markdown("This tool estimates house prices in Ames, Iowa, and helps you plan your mortgage.")

# --- SIDEBAR: The Inputs ---
st.sidebar.header("1. Property Details")

# Core Features (User controls these)
overall_qual = st.sidebar.slider("💎 Overall Quality (1-10)", 1, 10, 6)
gr_liv_area = st.sidebar.number_input("📐 Living Area (sq ft)", 500, 5000, 1500)
year_built = st.sidebar.slider("🏗️ Year Built", 1870, 2010, 2000)

with st.sidebar.expander("➕ Advanced House Features"):
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 3000, 800)
    garage_cars = st.selectbox("Garage Capacity", [0, 1, 2, 3, 4], index=2)
    full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=2)
    lot_area = st.number_input("Lot Area (sq ft)", 1000, 50000, 9600)
    neighborhood = st.selectbox("Neighborhood", ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt"])

# --- MAIN AREA: Tabs ---
tab1, tab2 = st.tabs(["🤖 Price Prediction", "💰 Mortgage Calculator"])

# --- TAB 1: AI Prediction ---
with tab1:
    st.subheader("Estimate Market Value")
    
    # --- CRITICAL FIX: The Full Payload ---
    # This dictionary must contain EVERY column the model expects.
    # We use user variables for dynamic fields and 'safe defaults' for the rest.
    house_data = {
        # --- User Dynamic Inputs ---
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "YearBuilt": year_built,
        "YearRemodAdd": year_built, # Assume remodel same as build
        "TotalBsmtSF": total_bsmt_sf,
        "GarageCars": garage_cars,
        "GarageArea": garage_cars * 250, # Estimate area based on cars
        "FullBath": full_bath,
        "LotArea": lot_area,
        "Neighborhood": neighborhood,
        "1stFlrSF": gr_liv_area,      # Simplified assumption
        
        # --- Static Defaults (Required by Model Pipeline) ---
        "MSSubClass": 20, "MSZoning": "RL", "LotFrontage": 80.0, 
        "Street": "Pave", "Alley": "None", "LotShape": "Reg", "LandContour": "Lvl",
        "Utilities": "AllPub", "LotConfig": "Inside", "LandSlope": "Gtl", 
        "Condition1": "Norm", "Condition2": "Norm", "BldgType": "1Fam", "HouseStyle": "1Story",
        "OverallCond": 5, "RoofStyle": "Gable", "RoofMatl": "CompShg", 
        "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd", "MasVnrType": "None", "MasVnrArea": 0.0, 
        "ExterQual": "TA", "ExterCond": "TA", "Foundation": "PConc", 
        "BsmtQual": "Gd", "BsmtCond": "TA", "BsmtExposure": "No", 
        "BsmtFinType1": "GLQ", "BsmtFinSF1": 0.0, 
        "BsmtFinType2": "Unf", "BsmtFinSF2": 0.0, "BsmtUnfSF": 0.0, 
        "Heating": "GasA", "HeatingQC": "Ex", "CentralAir": "Y", "Electrical": "SBrkr", 
        "2ndFlrSF": 0, "LowQualFinSF": 0, 
        "BsmtFullBath": 0.0, "BsmtHalfBath": 0.0, "HalfBath": 0, 
        "BedroomAbvGr": 3, "KitchenAbvGr": 1, "KitchenQual": "Gd", 
        "TotRmsAbvGrd": 6, "Functional": "Typ", "Fireplaces": 1, "FireplaceQu": "Gd", 
        "GarageType": "Attchd", "GarageYrBlt": year_built, "GarageFinish": "RFn", 
        "GarageQual": "TA", "GarageCond": "TA", "PavedDrive": "Y", 
        "WoodDeckSF": 0, "OpenPorchSF": 0, "EnclosedPorch": 0, 
        "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0, "PoolQC": "None", 
        "Fence": "None", "MiscFeature": "None", "MiscVal": 0, 
        "MoSold": 6, "YrSold": 2025, "SaleType": "WD", "SaleCondition": "Normal"
    }

    if st.button("🚀 Predict Price", type="primary"):
        # Ensure this matches your Docker port (5001 based on your success logs)
        api_url = "http://127.0.0.1:5000/predict" 
        
        try:
            with st.spinner("Analyzing market data..."):
                response = requests.post(api_url, json=house_data)
            
            if response.status_code == 200:
                prediction = response.json().get("predicted_price")
                
                # Display Result
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"### Estimated Price: ${prediction:,.0f}")
                with col2:
                    price_per_sqft = prediction / gr_liv_area
                    st.metric("Price per Sq. Ft.", f"${price_per_sqft:,.2f}")
                
                st.session_state['last_prediction'] = prediction
            else:
                st.error(f"API Error: {response.text}")
                
        except Exception as e:
            st.error("⚠️ Connection Failed. Is your Docker container running?")
            st.caption(f"Error details: {e}")

# --- TAB 2: Mortgage Calculator ---
with tab2:
    st.subheader("Plan Your Payments")
    default_price = st.session_state.get('last_prediction', 150000.0)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        home_price = st.number_input("Home Price ($)", value=int(default_price))
    with col_b:
        down_payment_pct = st.number_input("Down Payment (%)", 0, 100, 20)
    with col_c:
        interest_rate = st.number_input("Interest Rate (%)", 0.0, 15.0, 6.5)
        
    years = st.slider("Loan Term (Years)", 10, 30, 30)
    
    loan_amount = home_price * (1 - down_payment_pct/100)
    monthly_rate = (interest_rate / 100) / 12
    num_payments = years * 12
    
    if monthly_rate > 0:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    else:
        monthly_payment = loan_amount / num_payments
        
    st.divider()
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Monthly Payment", f"${monthly_payment:,.2f}")
    m_col2.metric("Loan Amount", f"${loan_amount:,.0f}")
    m_col3.metric("Total Interest", f"${(monthly_payment * num_payments) - loan_amount:,.0f}")