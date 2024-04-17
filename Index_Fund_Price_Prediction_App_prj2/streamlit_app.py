# dependancies
#pip install yfinance
#pip install streamlit

#import streamlit as st
#from datetime import date 
#import pandas as pd
#import yfinance as yf
# import plotly as px
# from plotly import graph_objs as go

import subprocess
import sys

try:
    import yfinance
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Index Fund Price Forecast App')

stocks = ('VOO', 'SPY', 'IVV', 'SWPPX', 'QQQ', 'VTWO', 'VTSAX', 'DIA')
selected_stock = st.selectbox('Select dataset for prediction', stocks)
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache

def load_index_fund_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data= load_index_fund_data(selected_stock)
data_load_state.text('Loading data... Finito!')

st.subheader('Raw data')
st.write(data.tail())

#Plot the raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
        
plot_raw_data()
