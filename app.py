import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Page setup
st.set_page_config(page_title="Financial ML App", layout="centered")
st.title("📈 Interactive Financial ML Application")

# Welcome Message and GIF
st.image("200w (1).gif", caption="Welcome to the Financial ML App")
st.markdown("""
This app allows you to upload a financial dataset or fetch stock data, preprocess it, apply machine learning, and visualize results step-by-step.
""")

# SidebarS
st.sidebar.title("🔍 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Kaagle Dataset (CSV)", type=["csv"])
ticker = st.sidebar.text_input("Enter Stock Ticker for Yahoo Finance", value="AAPL")
fetch_data = st.sidebar.button("Fetch Data from Yahoo Finance")

# Load Data Step
if st.button("Step 1: Load Data"):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Kragle dataset loaded successfully!")
    elif fetch_data:
        df = yf.download(ticker, start="2020-01-01", end="2024-12-31")
        st.success(f"Yahoo Finance data for {ticker} fetched successfully!")
    else:
        st.warning("Please upload a dataset or fetch stock data.")
        df = None

    if df is not None:
        st.dataframe(df.head())

if 'df' not in st.session_state:
    st.session_state.df = None

# Preprocessing Step
if st.button("Step 2: Preprocess Data"):
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.dropna()

    if uploaded_file or fetch_data:
        st.session_state.df = df
        st.info(f"After preprocessing, data shape is: {df.shape}")
    else:
        st.warning("Load data first.")

# Feature Engineering
if st.button("Step 3: Feature Engineering"):
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        if 'Close' in df.columns:
            df['Return'] = df['Close'].pct_change()
            df.dropna(inplace=True)
            st.session_state.df = df
            st.success("Feature 'Return' added.")
            st.line_chart(df['Return'])
        else:
            st.warning("'Close' column not found for feature engineering.")
    else:
        st.warning("Please preprocess the data first.")

# Train/Test Split
if 'df' in st.session_state and st.session_state.df is not None:
    if 'Return' in st.session_state.df.columns:
        df = st.session_state.df.copy()
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['Return'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        pie_data = pd.DataFrame({"Set": ["Train", "Test"], "Size": [len(X_train), len(X_test)]})
        fig = px.pie(pie_data, names='Set', values='Size', title='Train/Test Split')
        st.plotly_chart(fig)
else:
    st.warning("Please generate features first.")


# Model Training
if st.button("Step 5: Train Model"):
    if 'X_train' in st.session_state:
        model = LinearRegression()
        model.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.model = model
        st.success("Model trained successfully!")
    else:
        st.warning("Please split the data first.")

# Evaluation
if st.button("Step 6: Evaluate Model"):
    if 'model' in st.session_state:
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        r2 = r2_score(st.session_state.y_test, y_pred)
        mse = mean_squared_error(st.session_state.y_test, y_pred)
        st.metric("R² Score", round(r2, 4))
        st.metric("Mean Squared Error", round(mse, 6))
        st.line_chart(y_pred)
    else:
        st.warning("Please train the model first.")

# Final Visualization
if st.button("Step 7: Visualize Predictions"):
    if 'model' in st.session_state:
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        results_df = pd.DataFrame({"Actual": st.session_state.y_test, "Predicted": y_pred})
        st.line_chart(results_df)
    else:
        st.warning("Please train and evaluate the model first.")
