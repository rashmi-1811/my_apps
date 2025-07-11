import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("📊 Financial Trend Analysis & Forecast Dashboard")

# --- File upload ---
uploaded_file = st.file_uploader("Upload your CSV file (must have a 'Date' column)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'], dayfirst=True, infer_datetime_format=True)
    df = df.sort_values('Date')
    st.success("✅ Data successfully loaded!")

    # --- Select columns ---
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    object_cols = df.select_dtypes(include='object').columns.tolist()
    all_cols = df.columns.tolist()

    # --- Sidebar Filters ---
    st.sidebar.header("🔎 Filters")

    # Date range
    min_date, max_date = df['Date'].min(), df['Date'].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    df_filtered = df.copy()
    if len(date_range) == 2:
        df_filtered = df[(df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))]

    # Category filter
    if object_cols:
        cat_col = st.sidebar.selectbox("Filter by Category Column (optional):", ["None"] + object_cols)
        if cat_col != "None":
            cat_values = st.sidebar.multiselect(f"Select {cat_col} values", options=df_filtered[cat_col].unique(), default=df_filtered[cat_col].unique())
            df_filtered = df_filtered[df_filtered[cat_col].isin(cat_values)]

    # Numeric column filter
    if numeric_cols:
        num_col = st.sidebar.selectbox("Filter by Numeric Column (optional):", ["None"] + numeric_cols)
        if num_col != "None":
            min_num, max_num = float(df_filtered[num_col].min()), float(df_filtered[num_col].max())
            range_num = st.sidebar.slider(f"Range for {num_col}", min_value=min_num, max_value=max_num, value=(min_num, max_num))
            df_filtered = df_filtered[(df_filtered[num_col] >= range_num[0]) & (df_filtered[num_col] <= range_num[1])]

    # --- Trend plot ---
    st.subheader("📈 Trend Analysis")

    if numeric_cols:
        price_column = st.selectbox("Select column to analyze:", numeric_cols)
        fig_line = px.line(df_filtered, x='Date', y=price_column, title=f"{price_column} Over Time", markers=True)
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("⚠️ No numeric columns found to plot.")

    # --- Monthly distribution ---
    st.subheader("📊 Monthly Distribution")
    if numeric_cols and not df_filtered.empty:
        df_filtered['Month'] = df_filtered['Date'].dt.strftime('%b')
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_data = df_filtered.groupby('Month')[price_column].sum().reindex(month_order).reset_index()

        chart_type = st.radio("Chart Type", ['Bar', 'Pie'], horizontal=True)
        if chart_type == 'Bar':
            fig_bar = px.bar(monthly_data, x='Month', y=price_column, title=f"{price_column} Distribution by Month")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            fig_pie = px.pie(monthly_data, names='Month', values=price_column, title=f"{price_column} Distribution by Month")
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("ℹ️ No data available for monthly distribution plot.")

    # --- Forecasting ---
    st.subheader("🔮 Forecasting with Prediction Intervals")

    if numeric_cols and not df_filtered.empty:
        forecast_model_

