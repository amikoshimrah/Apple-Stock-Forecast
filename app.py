import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Apple Forecasting", layout="wide")

# GitHub raw CSV URL
CSV_URL = "https://raw.githubusercontent.com/amikoshimrah/Apple-Stock-Forecast/main/AAPL.csv"

# Load historical data from GitHub
@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv(CSV_URL)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        df = df[df.index >= pd.to_datetime("2009-01-01")]  # Filter from 2009 onward
        return df[['Close']].dropna()
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return pd.DataFrame()

# Load models with associated last date
@st.cache_resource
def load_models():
    try:
        with open("arima_model_aapl.pkl", "rb") as f_arima:
            arima_model, arima_last_date = pickle.load(f_arima)
        with open("sarima_model_aapl.pkl", "rb") as f_sarima:
            sarima_model, sarima_last_date = pickle.load(f_sarima)
        return {
            "ARIMA": (arima_model, arima_last_date),
            "SARIMA": (sarima_model, sarima_last_date)
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}

# Load everything
historical_df = load_historical_data()
models = load_models()

# Sidebar controls
with st.sidebar:
    st.header("📊 Forecast Settings")
    model_choice = st.selectbox("Select Model", list(models.keys()) if models else [])
    forecast_months = st.slider("Forecast Period (months)", 1, 36, 12)
    st.write("🔵 Actual: Historical Apple Close Price")
    st.write("🔴 Forecast: Model Prediction")

# Title
st.title("📈 Apple Stock Price Forecasting")

if model_choice:
    st.markdown(f"Forecasting with **{model_choice}** model")

    # Forecast generation
    if st.button("🔮 Generate Forecast"):
        model, last_date = models[model_choice]

        try:
            forecast = model.forecast(steps=forecast_months)
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            st.stop()

        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(days=1),
            periods=forecast_months,
            freq='M'
        )

        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast}).set_index('Date')

        # Plot actual + forecast
        st.subheader("📉 Apple Stock Price Forecast")
        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(historical_df.index, historical_df['Close'], label='Actual (Historical)', color='blue')
        ax.plot(forecast_df.index, forecast_df['Forecast'], label=f'{model_choice} Forecast', color='red')

        ax.set_title(f"{model_choice} Forecast for Next {forecast_months} Months")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Forecast Table (formatted)
        st.subheader("🔍 Forecast Table")
        st.dataframe(forecast_df.style.format({"Forecast": "${:,.2f}"}))

# Optional Historical Table
if not historical_df.empty:
    with st.expander("📜 View Historical Data"):
        st.dataframe(historical_df.tail(50).style.format({"Close": "${:,.2f}"}))
