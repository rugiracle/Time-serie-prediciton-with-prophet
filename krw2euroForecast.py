"""
Korean won euro prediction using Prophet
historical exchange data can be downloaded from https://www.investing.com/currencies/eur-krw-historical-data
"""
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import os

st.title("KoreanWon Euro currency exchange rate prediction App")

n_years = st.slider("Years of prediction", 1, 2)
period = n_years * 365


def load_data():
    # historical exchange data downloaded from https://www.investing.com/currencies/eur-krw-historical-data
    euro_to_krw = pd.read_csv('EUR_KRW_Historical_Data.csv')
    return euro_to_krw


def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Price'], name='Price'))
    fig.layout.update(title_text="Time Series Data[EURO2KRW]", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


data_load_state = st.text("Load data...")
data = load_data()
data_load_state.text('Loading data...done')

st.subheader('Raw data')
st.write(data.head())

# data preprocessing
df_train = data[['Date','Price']]
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_train['Price'].replace(',','', regex=True, inplace=True)  # remove comma from price
df_train['Price'] = pd.to_numeric(df_train['Price'], errors='coerce')  # convert as numeric values were stored as string
df_train.sort_values(by=['Date'], inplace=True, ascending=True)

plot_raw_data(df_train)

df_train = df_train.rename(columns={"Date": "ds", "Price": "y"})  # rename columns for Prophet

# Predict euro price with Prophet.
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot prediction
st.subheader('Forecast data')
st.write(forecast.tail())

fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
#os._exit(0)
#st.stop()




