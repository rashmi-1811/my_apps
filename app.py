import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# --- Load Data ---

df = pd.read_csv("List of Orders.csv")
df1 = pd.read_csv("Order Details.csv")

dfn = df.dropna().copy()
dfn['Order Date'] = pd.to_datetime(dfn['Order Date'], format='%d-%m-%Y')
dfn['Order_month'] = dfn['Order Date'].dt.month_name()
dfn['Order_year'] = dfn['Order Date'].dt.year

merged_df = pd.merge(dfn, df1, on=['Order ID'], how='inner')
merged_df['Month_num'] = pd.to_datetime(merged_df['Order_month'], format='%B').dt.month
merged_df['Year_month'] = merged_df['Order_year'].astype(str) + '-' + merged_df['Order_month']

# --- App Setup ---

app = Dash(__name__)

# --- Layout ---

app.layout = html.Div([
    html.H1("Advanced Interactive Sales Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Select State"),
        dcc.Dropdown(
            options=[{'label': s, 'value': s} for s in merged_df['State'].unique()],
            value=None,
            multi=True,
            id='state_filter'
        ),
        html.Label("Select Category"),
        dcc.Dropdown(
            options=[{'label': c, 'value': c} for c in merged_df['Category'].unique()],
            value=None,
            multi=True,
            id='category_filter'
        ),
        html.Label("Select Year"),
        dcc.Dropdown(
            options=[{'label': y, 'value': y} for y in merged_df['Order_year'].unique()],
            value=None,
            multi=True,
            id='year_filter'
        ),
    ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        html.H3(id='total_profit_kpi', style={'color': 'green'}),
        html.H3(id='total_amount_kpi', style={'color': 'blue'}),
        html.H3(id='total_orders_kpi', style={'color': 'orange'}),
    ], style={'width': '70%', 'display': 'inline-block', 'paddingLeft': '50px'}),

    dcc.Graph(id='category_pie'),
    dcc.Graph(id='trend_line'),
    dcc.Graph(id='city_heatmap'),
    dcc.Graph(id='bubble_scatter'),
])

# --- Callbacks ---

@app.callback(
    [
        Output('category_pie', 'figure'),
        Output('trend_line', 'figure'),
        Output('city_heatmap', 'figure'),
        Output('bubble_scatter', 'figure'),
        Output('total_profit_kpi', 'children'),
        Output('total_amount_kpi', 'children'),
        Output('total_orders_kpi', 'children'),
    ],
    [
        Input('state_filter', 'value'),
        Input('category_filter', 'value'),
        Input('year_filter', 'value')
    ]
)
def update_charts(states, categories, years):
    dff = merged_df.copy()

    if states:
        dff = dff[dff['State'].isin(states)]
    if categories:
        dff = dff[dff['Category'].isin(categories)]
    if years:
        dff = dff[dff['Order_year'].isin(years)]

    # KPIs
    total_profit = dff['Profit'].sum()
    total_amount = dff['Amount'].sum()
    total_orders = dff['Order ID'].nunique()

    # Pie chart for Category
    pie_fig = px.pie(
        dff.groupby('Category')['Profit'].sum().reset_index(),
        names='Category', values='Profit',
        title="Profit Distribution by Category",
        hole=0.4
    )
    pie_fig.update_traces(textinfo='percent+value', pull=[0.05]*len(dff['Category'].unique()))

    # Trend line
    trend_df = dff.groupby(['Order_year', 'Order_month']).agg({'Profit': 'sum', 'Amount': 'sum'}).reset_index()
    trend_df['Month_num'] = pd.to_datetime(trend_df['Order_month'], format='%B').dt.month
    trend_df = trend_df.sort_values(by=['Order_year', 'Month_num'])
    trend_df['Year_month'] = trend_df['Order_year'].astype(str) + '-' + trend_df['Order_month']

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=trend_df['Year_month'], y=trend_df['Profit'],
                                   name='Profit', mode='lines+markers', line=dict(color='green')))
    fig_trend.add_trace(go.Scatter(x=trend_df['Year_month'], y=trend_df['Amount'],
                                   name='Amount', mode='lines+markers', line=dict(color='blue'), yaxis='y2'))

    fig_trend.update_layout(
        title="Profit and Amount Trend Over Months",
        xaxis_title="Month",
        yaxis=dict(title="Profit"),
        yaxis2=dict(title="Amount", overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99)
    )

    # City heat map
    city_df = dff.groupby('City')['Profit'].sum().reset_index()
    heat_fig = px.density_mapbox(
        dff, lat='Latitude', lon='Longitude', z='Profit',
        radius=20, center=dict(lat=dff['Latitude'].mean(), lon=dff['Longitude'].mean()),
        mapbox_style="open-street-map", title="Profit Distribution by City"
    ) if 'Latitude' in dff.columns and 'Longitude' in dff.columns else px.bar(city_df, x='City', y='Profit', title="City-wise Profit Distribution")

    # Bubble scatter
    scatter_fig = px.scatter(
        dff,
        x='Profit', y='Amount',
        size='Quantity', color='City',
        hover_name='Order ID',
        animation_frame='Order_month',
        title="Profit vs Amount Bubble Scatter by City",
        template='plotly_white'
    )
    scatter_fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    scatter_fig.update_layout(legend_title="City")

    # KPI texts
    kpi1 = f"Total Profit: ${total_profit:,.0f}"
    kpi2 = f"Total Amount: ${total_amount:,.0f}"
    kpi3 = f"Total Orders: {total_orders}"

    return pie_fig, fig_trend, heat_fig, scatter_fig, kpi1, kpi2, kpi3

# --- Run App ---

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=8080)
