# Bitcoin Price Prediction using XGBoost

This project uses XGBoost, a powerful machine learning algorithm, to predict Bitcoin prices based on historical data. The model uses the past 5 days of Bitcoin prices to predict the next day's price.

## Prerequisites

Before running this script, make sure you have the following Python packages installed:

```bash
pip install xgboost
pip install numpy
pip install pandas
pip install scikit-learn
pip install pycoingecko
pip install matplotlib
```

## How to Use

1. **Setup**
   - Ensure you have Python 3.x installed on your system
   - Install all required dependencies using the commands above
   - Make sure you have an active internet connection (required for fetching Bitcoin data)

2. **Running the Script**
   - Simply run the script using Python:
   ```bash
   python xgboost_bitcoin.py
   ```

3. **What the Script Does**
   - Fetches the last 365 days of Bitcoin price data from CoinGecko API
   - Creates features using the past 5 days of price data
   - Splits the data into training (80%) and testing (20%) sets
   - Trains an XGBoost model on the historical data
   - Makes predictions on the test set
   - Displays two plots:
     - Feature importance plot showing which historical prices are most influential
     - Actual vs. predicted Bitcoin prices over time

4. **Output**
   - The script will print the Mean Squared Error (MSE) of the predictions
   - Two plots will be displayed:
     - A bar chart showing the importance of each feature
     - A line chart comparing actual vs. predicted Bitcoin prices

## Notes

- The script uses the free CoinGecko API, which has rate limits
- The model uses a simple feature set (past 5 days of prices) and could be enhanced with additional features
- The predictions are based on historical data and should not be used as financial advice
- The model's performance can be improved by tuning hyperparameters or adding more features

## Limitations

- The free CoinGecko API has rate limits and only provides 365 days of historical data
- The model is relatively simple and doesn't account for external factors affecting Bitcoin prices
- Predictions are based solely on historical price data 