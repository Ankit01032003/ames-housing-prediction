import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
import os

# Page configuration
st.set_page_config(
    page_title="🏠 Ames House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# Load model and preprocessing objects
@st.cache_resource
def load_artifacts():
    try:
        model = load_model('models/ann_model.h5')
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Load data for visualizations
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/AmesHousing.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Title and description
st.title("🏠 Ames Housing Price Predictor")
st.markdown("### Predict house prices using Artificial Neural Network")
st.markdown("---")

# Load artifacts
model, scaler, features = load_artifacts()
df = load_data()

# Check if everything is loaded properly
if model is None or df is None:
    st.error("⚠️ Model files or dataset not found!")
    st.info("Please make sure you have:")
    st.write("1. Trained the model using `python train_model.py`")
    st.write("2. Placed `AmesHousing.csv` in the `data/` folder")
    st.write("3. The `models/` folder contains `ann_model.h5`, `scaler.pkl`, and `feature_names.pkl`")
    st.stop()

# Sidebar for navigation
st.sidebar.title("📋 Navigation")
option = st.sidebar.radio(
    "Choose an option:",
    ["🔮 Predict Price", "📊 Data Overview", "📈 Model Performance"]
)

if option == "🔮 Predict Price":
    st.header("Enter House Features")
    st.markdown("Fill in the details below to get a price prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏗️ Basic Features")
        overall_qual = st.slider("Overall Quality", 1, 10, 6, help="1=Very Poor, 10=Very Excellent")
        overall_cond = st.slider("Overall Condition", 1, 9, 5, help="1=Very Poor, 9=Very Excellent")
        year_built = st.number_input("Year Built", 1872, 2010, 2000)
        year_remod = st.number_input("Year Remod/Add", 1950, 2010, 2005)
        gr_liv_area = st.number_input("Above Grade Living Area (sq ft)", 300, 5000, 1500)
        
    with col2:
        st.subheader("🛏️ Rooms & Baths")
        bedrooms = st.slider("Bedrooms", 0, 8, 3)
        full_bath = st.slider("Full Bathrooms", 0, 4, 2)
        half_bath = st.slider("Half Bathrooms", 0, 2, 1)
        kitchen_abvgr = st.slider("Kitchens", 0, 3, 1)
        tot_rms_abv_grd = st.slider("Total Rooms Above Ground", 2, 14, 7)
        
    with col3:
        st.subheader("🏡 Area Features")
        lot_area = st.number_input("Lot Area (sq ft)", 1300, 215245, 9000)
        total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 6000, 1000)
        first_flr_sf = st.number_input("1st Floor Area (sq ft)", 300, 4000, 1000)
        second_flr_sf = st.number_input("2nd Floor Area (sq ft)", 0, 2000, 500)
        garage_area = st.number_input("Garage Area (sq ft)", 0, 1500, 500)
    
    # Prepare input data
    input_data = pd.DataFrame({
        'Overall Qual': [overall_qual],
        'Overall Cond': [overall_cond],
        'Year Built': [year_built],
        'Year Remod/Add': [year_remod],
        'Gr Liv Area': [gr_liv_area],
        'Bedroom AbvGr': [bedrooms],
        'Full Bath': [full_bath],
        'Half Bath': [half_bath],
        'Kitchen AbvGr': [kitchen_abvgr],
        'TotRms AbvGrd': [tot_rms_abv_grd],
        'Lot Area': [lot_area],
        'Total Bsmt SF': [total_bsmt_sf],
        '1st Flr SF': [first_flr_sf],
        '2nd Flr SF': [second_flr_sf],
        'Garage Area': [garage_area]
    })
    
    # Make prediction button
    if st.button("🔮 Predict Sale Price", type="primary"):
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        predicted_price = prediction[0][0]
        
        # Display results
        st.markdown("---")
        st.header("📊 Prediction Result")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style='background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center;'>
                <h2 style='color: white;'>${predicted_price:,.2f}</h2>
                <p style='color: white;'>Predicted Sale Price</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show similar houses from dataset
        st.subheader("🏘️ Similar Houses in Dataset")
        similar_houses = df[
            (df['Overall Qual'].between(overall_qual-1, overall_qual+1)) &
            (df['Gr Liv Area'].between(gr_liv_area-200, gr_liv_area+200))
        ].head(5)
        
        if not similar_houses.empty:
            st.dataframe(similar_houses[['Overall Qual', 'Gr Liv Area', 'SalePrice']])
        else:
            st.info("No similar houses found in the dataset")

elif option == "📊 Data Overview":
    st.header("📊 Dataset Overview")
    
    tab1, tab2, tab3 = st.tabs(["📋 Basic Info", "📈 Visualizations", "📉 Statistics"])
    
    with tab1:
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        st.subheader("First 5 Rows")
        st.dataframe(df.head())
        
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            fig = px.bar(
                x=missing.index,
                y=missing.values,
                title="Missing Values by Column",
                labels={'x': 'Columns', 'y': 'Count of Missing Values'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing values found!")
    
    with tab2:
        st.subheader("Sale Price Distribution")
        fig = px.histogram(
            df, x='SalePrice', 
            nbins=50,
            title="Distribution of Sale Prices",
            color_discrete_sequence=['blue']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Correlation with Sale Price")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        corr = df[numeric_cols].corr()['SalePrice'].sort_values(ascending=False).head(10)
        fig = px.bar(
            x=corr.index,
            y=corr.values,
            title="Top 10 Features Correlated with Sale Price",
            labels={'x': 'Features', 'y': 'Correlation'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())

elif option == "📈 Model Performance":
    st.header("📈 Model Performance Metrics")
    
    # Sample metrics (you'll replace with actual values)
    metrics = {
        'R² Score': 0.88,
        'RMSE': 28600,
        'MAE': 19500,
        'MAPE': 12.5
    }
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{metrics['R² Score']:.3f}")
    with col2:
        st.metric("RMSE", f"${metrics['RMSE']:,.0f}")
    with col3:
        st.metric("MAE", f"${metrics['MAE']:,.0f}")
    with col4:
        st.metric("MAPE", f"{metrics['MAPE']:.1f}%")
    
    st.subheader("Actual vs Predicted Values")
    # Create sample scatter plot (replace with actual test data)
    actual = np.random.normal(180000, 50000, 100)
    predicted = actual + np.random.normal(0, 20000, 100)
    
    fig = px.scatter(
        x=actual,
        y=predicted,
        title="Actual vs Predicted Sale Price",
        labels={'x': 'Actual Price', 'y': 'Predicted Price'},
        trendline="ols"
    )
    fig.add_shape(
        type="line",
        x0=min(actual),
        y0=min(actual),
        x1=max(actual),
        y1=max(actual),
        line=dict(color="red", dash="dash")
    )
    st.plotly_chart(fig, use_container_width=True)