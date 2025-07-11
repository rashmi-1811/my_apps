import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("📈 Financial Trend Analysis & Forecast Dashboard")

uploaded_file = st.file_uploader("Upload your CSV file (must have a 'Date' column)", type=["csv"])


if uploaded_file is not None:
    try:
        # ⬇️ Parse and filter df
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')
        df['Year'] = df['Date'].dt.year

        for col in df.columns:
            if col not in ['Date', 'Year', 'Unnamed: 0']:
                df[col] = df[col].astype(str).str.replace(",", "", regex=False).replace("nan", pd.NA)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        st.sidebar.header("🔎 Filters")
        years = sorted(df['Year'].dropna().unique())
        selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)
        df = df[df['Year'].isin(selected_years)]

        if df.empty:
            st.warning("⚠️ No data available for the selected year(s). Please adjust your filters.")
            st.stop()

        st.write("✅ Filtered data shape:", df.shape)
        st.dataframe(df.head())

        st.success("✅ Data successfully loaded!")

        # --- 📊 Trend Visualizations ---
        st.subheader("Trend over Time")
        numeric_cols = [col for col in df.select_dtypes(include='number').columns if col not in ['Year', 'Unnamed: 0']]
        price_column = st.selectbox("Select Column to Analyze", numeric_cols)

        fig = px.line(df, x='Date', y=price_column, title=f"{price_column} Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        # --- 📅 Monthly Distribution ---
        st.subheader("Monthly Distribution")
        df['Month'] = df['Date'].dt.strftime('%b')
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_data = df.groupby('Month')[price_column].sum().reindex(month_order).reset_index()

        chart_type = st.radio("Select Chart Type", ['Bar', 'Pie'], horizontal=True)

        if chart_type == 'Pie':
            fig2 = px.pie(monthly_data, names='Month', values=price_column,
                          title=f"{price_column} Distribution by Month")
        else:
            fig2 = px.bar(monthly_data, x='Month', y=price_column,
                          title=f"{price_column} Distribution by Month")

            st.plotly_chart(fig2, use_container_width=True)

        
        # --- 🔮 Forecasting ---
        st.subheader("Forecasting with Prediction Intervals")

        forecast_model = st.selectbox("Choose Forecast Model", ['Holt-Winters', 'ARIMA'])
        forecast_days = st.slider("Forecast Days Ahead", 7, 90, 30)

        ts = df.set_index('Date')[price_column].asfreq('D')
        ts = ts.fillna(method='ffill')

        try:
            if forecast_model == 'Holt-Winters':
                model = ExponentialSmoothing(ts, trend='add', seasonal=None, initialization_method='estimated')
                fit = model.fit()
                forecast = fit.forecast(forecast_days)

                resid_std = np.std(fit.resid)
                upper = forecast + 1.96 * resid_std
                lower = forecast - 1.96 * resid_std

            else:  # ARIMA
                model = ARIMA(ts, order=(5, 1, 0))
                fit = model.fit()
                forecast = fit.forecast(forecast_days)

                conf_int = fit.get_forecast(steps=forecast_days).conf_int(alpha=0.05)
                lower = conf_int.iloc[:, 0]
                upper = conf_int.iloc[:, 1]

                forecast_df = pd.DataFrame({
                'Date': forecast.index,
                'Forecast': forecast,
                'Lower': lower,
                'Upper': upper
                }).reset_index(drop=True)

                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(
                x=ts.index, y=ts,
                mode='lines', name='Historical'
                ))
                fig_forecast.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Forecast'],
                mode='lines', name='Forecast', line=dict(color='red')
                ))
                fig_forecast.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Upper'],
                mode='lines', line=dict(width=0), showlegend=False
                ))
                fig_forecast.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Lower'],
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)',
                name='Confidence Interval'
                ))

                fig_forecast.update_layout(
                title='Forecast with Prediction Intervals',
                xaxis_title='Date', yaxis_title=price_column
                )

                st.plotly_chart(fig_forecast, use_container_width=True)
                st.success("✅ Forecasting complete!")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
    except Exception as e:
        st.error(f"⚠️ File processing error: {e}")

else:
    st.info("Please upload a CSV file to start.")
