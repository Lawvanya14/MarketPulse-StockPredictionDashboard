
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title='📈 MarketPulse - Stock Price Prediction', layout='wide')
st.title('📈 MarketPulse – Stock Price Prediction')
st.markdown('Predict stock trends using ensemble regression models (Random Forest, Gradient Boosting, and XGBoost).')

# --- Load CSV ---
csv_path = '/content/NFLX.csv'
df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df.sort_values('Date')

st.subheader('📊 Data Preview')
st.dataframe(df.head())

# --- Feature Engineering ---
df['Prev_Close'] = df['Close'].shift(1)
df['Price_Change'] = df['Close'] - df['Prev_Close']
df.dropna(inplace=True)

X = df[['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'Price_Change']]
y = df['Close']

# --- Split the data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train models ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, objective='reg:squarederror')

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# --- Ensemble prediction ---
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)
xgb_pred = xgb.predict(X_test)
ensemble_pred = (rf_pred + gb_pred + xgb_pred) / 3

# --- Evaluation ---
mse = mean_squared_error(y_test, ensemble_pred)
r2 = r2_score(y_test, ensemble_pred)

st.subheader('📈 Model Performance')
st.write('Mean Squared Error:', round(mse, 2))
st.write('R² Score:', round(r2, 4))

# --- Plot Results ---
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(y_test.values[:100], label='Actual', linewidth=2)
ax.plot(ensemble_pred[:100], label='Predicted', linestyle='dashed', linewidth=2)
ax.legend()
ax.set_title('Actual vs Predicted Stock Prices')
st.pyplot(fig)

# --- Sidebar for Custom Prediction ---
st.sidebar.header('🔮 Custom Prediction')
open_price = st.sidebar.number_input('Open', min_value=0.0, value=500.0)
high = st.sidebar.number_input('High', min_value=0.0, value=510.0)
low = st.sidebar.number_input('Low', min_value=0.0, value=495.0)
volume = st.sidebar.number_input('Volume', min_value=0, value=3000000)
prev_close = st.sidebar.number_input('Previous Close', min_value=0.0, value=502.0)
price_change = st.sidebar.number_input('Price Change', min_value=-50.0, value=3.0)

if st.sidebar.button('Predict Next Close Price'):
    custom_data = np.array([[open_price, high, low, volume, prev_close, price_change]])
    prediction = (rf.predict(custom_data)[0] + gb.predict(custom_data)[0] + xgb.predict(custom_data)[0]) / 3
    st.sidebar.success(f'Predicted Close Price: ${round(prediction, 2)}')

pred_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': ensemble_pred})
st.download_button('📥 Download Predictions as CSV', data=pred_df.to_csv(index=False),
                   file_name='predictions.csv', mime='text/csv')

st.write('---')
st.caption('Developed by Codiee Lawv 2.0 – Data Analyst & ML Enthusiast')
st.title('📈 MarketPulse – Stock Price Prediction')
st.markdown('Predict stock trends using ensemble regression models (Random Forest, Gradient Boosting, and XGBoost).')
st.info("Predictions are generated using an ensemble of Random Forest, Gradient Boosting, and XGBoost models.")
