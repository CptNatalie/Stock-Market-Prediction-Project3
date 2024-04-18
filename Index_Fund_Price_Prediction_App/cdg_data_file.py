import os
import json
import requests
import numpy as np 
import pandas as pd
from sklearn.svm import SVR
from datetime import timedelta, datetime
import plotly.graph_objs as go
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.datasets import make_regression, make_swiss_roll
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


#building the url with the API Key needed to 
def urlBuilder(ticker, functionType, outputSize):
    load_dotenv('apiKeys.env')
    api_key = os.getenv("extraKey")
    url = "https://www.alphavantage.co/query?"
    if(functionType != None):
        url = url + f"function={functionType}"
    if(ticker != None):
        if(functionType == "NEWS_SENTIMENT"):
            url = url + f"&tickers={ticker}"
        else:
            url = url + f"&symbol={ticker}"
    if(outputSize != None):
        url = url + f"&outputsize={outputSize}"
    url = url + f"&apikey={api_key}"
    print(url)
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

# Prediction Function
def regression_predictions(df):
    X=df.drop(columns=['target','tomorrow','close','date'], axis=1)
    y1=df['close']
    X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.3, random_state=42)
    
    #fit training data
    model = LinearRegression()
    model.fit(X1_train,y1_train)
    
    # Make predictions on testing data
    y_pred = model.predict(X)
    
    df['y_pred_LR'] = y_pred
    
    model = ExtraTreesRegressor()
    model.fit(X1_train,y1_train)
    
    # Make predictions on testing data
    y_pred = model.predict(X)
    
    df['y_pred_ETR'] = y_pred
    
    model = RandomForestRegressor()
    model.fit(X1_train,y1_train)
    
    # Make predictions on testing data
    y_pred = model.predict(X)
    
    df['y_pred_RFG'] = y_pred
    
    model = AdaBoostRegressor()
    model.fit(X1_train,y1_train)
    
    # Make predictions on testing data
    y_pred = model.predict(X)
    
    df['y_pred_ABR'] = y_pred
    
    return df

def graph_preds(df, ticker):
    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['y_pred'], mode='lines', name='Predicted Price'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Actual Price'))
    # fig.add_trace(go.Scatter(x=df['date'], y=df['y_pred_LR'], mode='lines', name='LR model'))
    fig.update_layout(title=f'Stock Prices and Predictions for {ticker}',
                      xaxis_title='Date',
                      yaxis_title='Price')
    fig.show()
    
def preds_output(df):
    num_shares = int(input("Enter the number of shares owned: "))
    investment_amount = float(input("Enter how much you have to invest: "))
    trend = 'higher' if df['close'].iloc[-1] > df['y_pred'].iloc[-1] else 'lower'
    action = 'buy' if trend == 'higher' else 'sell'
    optimal_volume = min(investment_amount / df['y_pred'].iloc[-1], num_shares)
    investment_value = df['close'].iloc[-1] * num_shares  # This is speculative
    
    print(f"Future predicted price for tomorrow: {df['close'].iloc[-1]}")
    print(f"The trend for the future price is {trend}.")
    print(f"You should {action}.")
    print(f"The optimal volume to {action} is {optimal_volume} shares.")
    print(f"The speculative future value of your investment is: ${investment_value:.2f}.")
    print("Please note, this is a prediction based on historical data and carries risk.")
    
def save_plot_as_image(df, company):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Close Price'))
    fig.update_layout(title=f'Stock Prices for {company}',
                      xaxis_title='Date',
                      yaxis_title='Price')
    # Save the plot as an image
    image_file = f"Resources/outputs/plot_{company}.png"
    fig.write_image(image_file)
    print(f"Plot saved as {image_file}")

    
def run(company, ticker):
    
    file_path = f'Resources/alphavantage/{ticker.lower()}.csv'
    # Index_Fund_Price_Prediction_App/Resources/alphavantage/ibm.csv
    print(file_path)
    if os.path.exists(file_path):
        print("Pulling from local storage")
        df_stock = loadPreSavedData(file_path)
    else:
        print("Pulling new data from AlphaVantage")
        df_stock = loadData(ticker)
        df_news = loadNewsSentiments(ticker)    
        print(df_stock.head())
        print(df_news.head())
        
        # Merge the DataFrames on the 'date' column
        merged_df = pd.merge(df_stock, df_news, on='date', how='inner')
        # Display the merged DataFrame
        print(merged_df.tail())
        
        # Save the merged DataFrame as a CSV file with a specific path
        merged_df.to_csv(f'/Resources/alphavantage/{ticker.lower()}.csv', index=False)
    
    

if __name__ == "__main__":
    company = "Apple"
    ticker = input("Enter a stock ticker: ").upper()
    run(company, ticker)