import yfinance as yf

# Function to get the stock price
def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    price = stock.history(period="1d")['Close'].iloc[-1]
    return price

# Function to get the stock info
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info['longBusinessSummary']

# Basic dialogue handling
def handle_dialogue(user_input):
    if 'price of' in user_input:
        words = user_input.split()
        ticker = words[-1].upper()  # assuming the last word is the ticker symbol
        try:
            price = get_stock_price(ticker)
            return f"The current price of {ticker} is ${price:.2f}"
        except Exception as e:
            return f"Sorry, I couldn't find data for {ticker}."
    elif 'info about' in user_input:
        words = user_input.split()
        ticker = words[-1].upper()  # assuming the last word is the ticker symbol
        try:
            info = get_stock_info(ticker)
            return f"Here's some information about {ticker}: {info}"
        except Exception as e:
            return f"Sorry, I couldn't find information for {ticker}."
    else:
        return "I'm not sure how to respond to that. I can provide stock prices and information if you tell me the stock ticker."

# Main function to run the chatbot
def run_chatbot():
    print("Welcome to the Stock Chatbot. Ask me for the price of a stock or information about a company.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Stock Chatbot: Goodbye!")
            break
        response = handle_dialogue(user_input)
        print(f"Stock Chatbot: {response}")

# Run the chatbot
if __name__ == "__main__":
    run_chatbot()
