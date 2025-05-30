import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pycoingecko import CoinGeckoAPI
import matplotlib.pyplot as plt
# import datetime # No longer strictly needed for this revised approach

# Initialize CoinGecko API
cg = CoinGeckoAPI()

# Get Bitcoin historical data for the last 365 days
# Due to API limitations for free users (max 365 days for ranged queries),
# we revert to fetching the last 365 days of data.
bitcoin_data = cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='usd', days=365)

# Convert data to Pandas DataFrame
prices = bitcoin_data['prices']
df = pd.DataFrame(prices, columns=['timestamp', 'price'])

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Feature Engineering (using lagged prices as features)
for i in range(1, 6):  # Using past 5 days' prices as features
    df[f'price_lag_{i}'] = df['price'].shift(i)

df.dropna(inplace=True)

# Define features (X) and target (y)
X = df[[f'price_lag_{i}' for i in range(1, 6)]]
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Initialize and train the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(model, ax=ax)
plt.title('Feature Importance')
plt.show()

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Prices')
plt.plot(y_test.index, predictions, label='Predicted Prices', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price (USD)')
plt.title('Bitcoin Price Prediction with XGBoost')
plt.legend()
plt.show() 