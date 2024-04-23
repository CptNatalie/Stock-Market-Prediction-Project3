import os
import json
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
import plotly.graph_objs as go
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras.layers import Dense, LSTM, Input
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#building the url with the API Key needed to 
def urlBuilder(ticker, functionType, outputSize):
    # Load the API key from the environment
    load_dotenv('apiKeys.env')
    api_key = os.getenv("extraKey")
    url = "https://www.alphavantage.co/query?"
    if(functionType != None):
        url = url + f"function={functionType}"
    # If the function is NEWS_SENTIMENT, use 'tickers' instead of 'symbol'
    if(ticker != None):
        if(functionType == "NEWS_SENTIMENT"):
            url = url + f"&tickers={ticker}&limit=1000"
        else:
            url = url + f"&symbol={ticker}"
    # Add outputSize parameter to the URL if provided
    if(outputSize != None):
        url = url + f"&outputsize={outputSize}"
    url = url + f"&apikey={api_key}"
    # print(url)
    return url


#API call
def loadData(ticker):
    functionType = 'TIME_SERIES_DAILY'
    interval = 'Daily'
    outputsize = 'full'
    url = urlBuilder(ticker, functionType, outputsize)
    # Get the response
    response = requests.get(url)
    data = response.json()
    # Extract time series data
    response = requests.get(url)
    data = response.json()

    # Extract time series data
    time_series_data = data.get('Time Series (Daily)', {})

    # Convert to DataFrame
    df = pd.DataFrame(time_series_data).T

    # Rename columns
    df.columns = ["open", "high", "low", "close", "volume"]

    # Convert data types
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": int})

    # Convert index to datetime
    df.index = pd.to_datetime(df.index)

    # Sort index
    df.sort_index(inplace=True)

    # Reset index
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    return df

def loadPreSavedData(file_path):
    # Reading in the CSV about the stock data from the local repo
    df = pd.read_csv(file_path)
    # Convert the 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])
    return df


def loadNewsSentiments(ticker):
    functionType = 'NEWS_SENTIMENT'
    url = urlBuilder(ticker, functionType, None)
    
    # Get the response
    response = requests.get(url)
    data = response.json()
    # Extract time series data
    news_feed = data.get(f'feed', {})

    # Convert to DataFrame
    df = pd.DataFrame(news_feed)
    df = df[['title', 'time_published', 'summary', 'overall_sentiment_score']]
    df['time_published'] = df['time_published'].str.split('T').str[0]
    # Convert the 'time_published' column to datetime
    df['time_published'] = pd.to_datetime(df['time_published'])

    # Format the date as YYYY-MMM-DD
    df['time_published'] = df['time_published'].dt.strftime('%Y-%m-%d')
    
    df_grouped = df.groupby('time_published').agg({
        'title': lambda x: ', '.join(x),
        'summary': lambda x: ', '.join(x),
        'overall_sentiment_score': 'mean'
    }).reset_index()
    
    df_grouped.columns = ["date", "title", "summary", "sentiment"]
    
    # Convert index to datetime
    df.index = pd.to_datetime(df.index)

    # Sort index
    df.sort_index(inplace=True)

    # Reset index
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    
    return df_grouped

#This will be how the data is split into training and test data
def preparing_data(df):
    df['RSI']=ta.rsi(df['close'], length=15)
    df['EMAF']=ta.ema(df['close'], length=20)
    df['EMAM']=ta.ema(df['close'], length=100)
    df['EMAS']=ta.ema(df['close'], length=150)
    df.dropna(subset=['EMAS','sentiment'], inplace=True)
    df['TargetNextClose'] = df['close'].shift(-1)
    X = df.drop(columns=['close','volume', 'date','TargetNextClose','summary', 'title'], axis=1)
    y = df['TargetNextClose']
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test, X, y

def neural_networking(X_train, X_test, y_train, y_test, X, y, df):
    # Reshape the input data for LSTM
    X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    X_reshaped = X.values.reshape((X.shape[0], 1, X.shape[1]))
    y_reshaped = y.values.reshape((y.shape[0], 1))
    
    # Create the input layer
    input_layer = Input(shape=(1, X_train.shape[1]), name='input_features')
    
    # Add LSTM layer
    lstm_layer = LSTM(150, name='first_layer')(input_layer)
    
    # Add Dense layer
    dense_layer = Dense(64, activation='relu')(lstm_layer)
    
    # Output layer
    output = Dense(1, activation='linear', name='output')(dense_layer)
        
    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    
    model.fit(
        X_train_reshaped,
        y_train,
        epochs=100,
        validation_split=0.2
    )
    test_results = model.evaluate(
        X_test_reshaped,
        y_test
    )
    print(test_results)
    # Scale y_train and y_test
    scaler = MinMaxScaler()
    scaler.fit(y_train.values.reshape(-1, 1))
    y_train_scaled = scaler.transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))
    
    y_pred_scaled = model.predict(X_reshaped)
    y_pred_flat = np.squeeze(y_pred_scaled)
    
    save_plot_as_image(df, y.values, y_pred_flat, ticker)

def save_plot_as_image(df, arr1, arr2, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=arr1, mode='lines', name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['TargetNextClose'], mode='lines', name='Predicted Price', line=dict(color='green')))
    fig.update_layout(title=f'Stock Prices for {ticker}',
                      xaxis_title='Date',
                      yaxis_title='Price')
    # Save the plot as an image
    image_file = f"Resources/outputs/plot_{ticker}.png"
    fig.write_image(image_file)
    print(f"Plot saved as {image_file}")
    
    
def run(ticker):
    
    file_path = f'Resources/alphavantage/{ticker.lower()}.csv'
    if os.path.exists(file_path):
        print("Pulling from local storage")
        merged_df = loadPreSavedData(file_path)
    else:
        print("Pulling new data from AlphaVantage")
        df_stock = loadData(ticker)
        df_news = loadNewsSentiments(ticker)
        # print(df_stock.head())
        # print(df_news.head())
        
        df_news['date'] = pd.to_datetime(df_news['date'])
        
        # Merge the DataFrames on the 'date' column
        merged_df = pd.merge(df_stock, df_news, on='date', how='left')
        
        # Save the merged DataFrame as a CSV file with a specific path
        merged_df.to_csv(file_path, index=False)
    # Display the merged DataFrame
    print(merged_df.head())
    print(merged_df.tail())
    
    X_train, X_test, y_train, y_test, X, y = preparing_data(merged_df)
    
    neural_networking(X_train, X_test, y_train, y_test, X, y, merged_df)


if __name__ == "__main__":
    ticker = input("Enter a stock ticker: ").upper()
    run(ticker)