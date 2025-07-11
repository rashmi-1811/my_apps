import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# --- 📄 Load Data ---
st.title("📈 Financial Trend Analysis & Forecast Dashboard")

uploaded_file = st.file_uploader(r"C:\Users\Dell\Downloads\csv work files\Stock Market Dataset.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    df = df.sort_values('Date')
    st.write("Data Preview", df.head())
    print(df['Apple_Price'].dtype)

    # --- 📊 EDA Visualizations ---
    st.subheader("Trend over time")

    price_column = st.selectbox("Select the Price Column", df.columns[2:])
    fig = px.line(df, x='Date', y=price_column, title=f"{price_column} over time")
    st.plotly_chart(fig, use_container_width=True)


    # --- 📅 Monthly Distribution ---
    st.subheader("Monthly Distribution")
    monthly_data = df.groupby('Date_Month')[price_column].sum().reset_index()
    monthly_data = monthly_data.sort_values('Date_Month')

    pie_or_bar = st.radio("Select chart type:", ['Pie', 'Bar'])

    if pie_or_bar == 'Pie':
        fig2 = px.pie(monthly_data, names='Date_Month', values=price_column, title=f"{price_column} Distribution by Month")
    else:
        fig2 = px.bar(monthly_data, x='Date_Month', y=price_column, title=f"{price_column} Distribution by Month")

    st.plotly_chart(fig2, use_container_width=True)

    # --- 🔮 Forecasting ---
    st.subheader("Forecasting")

    forecast_model = st.selectbox("Choose model", ['Holt-Winters', 'ARIMA'])
    forecast_days = st.slider("Forecast days ahead", 7, 90, 30)

    ts = df.set_index('Date')[price_column].asfreq('D')
    ts = ts.fillna(method='ffill')

    if forecast_model == 'Holt-Winters':
        model = ExponentialSmoothing(ts, trend='add', seasonal=None, initialization_method='estimated')
        fit = model.fit()
        forecast = fit.forecast(forecast_days)

        # Prediction intervals (approximation using residual std)
        resid_std = np.std(fit.resid)
        upper = forecast + 1.96 * resid_std
        lower = forecast - 1.96 * resid_std

    else:  # ARIMA
        model = ARIMA(ts, order=(5, 1, 0))
        fit = model.fit()
        forecast = fit.forecast(forecast_days)

        # Approximate intervals
        conf_int = fit.get_forecast(steps=forecast_days).conf_int(alpha=0.05)
        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]

    # --- 📈 Plot forecast ---
    forecast_df = pd.DataFrame({
        'Date': forecast.index,
        'Forecast': forecast,
        'Lower': lower,
        'Upper': upper
        }).reset_index(drop=True)

    fig_forecast = go.Figure()

    fig_forecast.add_trace(go.Scatter(
        x=ts.index,
        y=ts,
        mode='lines',
        name='Historical'
        ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='red')
        ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
          ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)',
        name='Confidence Interval'
        ))

    fig_forecast.update_layout(
        title='Forecast with Prediction Intervals',
        xaxis_title='Date',
        yaxis_title=price_column
        )

    st.plotly_chart(fig_forecast, use_container_width=True)

    st.success("✅ Forecasting complete!")

else:
    st.info("Please upload a CSV file to start.")
