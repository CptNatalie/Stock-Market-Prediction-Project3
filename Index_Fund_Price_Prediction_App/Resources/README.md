# Notes: 
## writing the code:
1. index funds to run ML models on: (Name:Ticker) these are the top rated index funds.
 - Fidelity ZERO Large Cap Index: FNILX
 - Vanguard S&P 500 ETF: VOO
 - SPDR S&P 500 ETF : SPY
 - iShares Core S&P 500 ETF: IVV
 - Schwab S&P Index Fund :SWPPX
 - Invesco QQQ Trust ETF: QQQ
 - Vanguard Russel 2000 ETF: VTWO
 - Vanguard Stock Market ETF :VTSAX
 - SPDR Dow Jones Industrial Average ETF Trust: DIA

 2. ticker data:
  - using the yfincance as a base code to pull ticker historical data.
  - pull data from Alpha Vantage
 
 3. convert pulled data into .csv  and save historical data
 
 4. create an API to store historical data, this will insure that we do no not hit the API pull limits from yfinance and AlphaVanguard
 
 5. pull data through Multi-model forecasting.

 6. pipeline design: (.py)
    1. define the function.
        - for all 10 mutual funds.
            a) pull historical data 
            b) convert to .csv
            c) create API
    2. Multi-model forecast
    3. call the function

 7.  user input function (.ipynb)
        a) user inputs desired parameters
        b) pipeline is run
        c) return predictions
        d) make suggestions based on predicted returns
