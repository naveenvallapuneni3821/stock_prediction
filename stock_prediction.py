import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

print("=" * 60)
print("📈 STOCK PRICE PREDICTION WITH LINEAR REGRESSION")
print("=" * 60)

print("\n📂 STEP 1: Loading your stock data...")

file_name = input("Enter your CSV file name (like 'stock_data.csv'): ")

try:
    df = pd.read_csv(file_name)
    print("✅ Data loaded successfully!")
except:
    print("❌ Couldn't find the file. Using sample data for demo...")
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 2)
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.randn(200) * 2,
        'High': prices + np.abs(np.random.randn(200) * 3),
        'Low': prices - np.abs(np.random.randn(200) * 3),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 200)
    })
    print("✅ Sample data created for demonstration!")

print("\n📊 STEP 2: Quick look at your data:")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

print("\n🛠️ STEP 3: Preparing data for prediction...")

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

print("Creating helpful features...")

df['Yesterday_Close'] = df['Close'].shift(1)
df['Price_Change'] = df['Close'] - df['Open']
df['Daily_Range'] = df['High'] - df['Low']
df['MA5'] = df['Close'].rolling(window=5).mean()

df = df.dropna()

print(f"✅ Data ready! Now we have {len(df)} rows after preparation")

print("\n🎯 STEP 4: Choosing what to predict...")

target = 'Close'
print(f"We'll predict the '{target}' price")

features = ['Yesterday_Close', 'Price_Change', 'Daily_Range', 'MA5']
if 'Volume' in df.columns:
    features.append('Volume')
if 'Open' in df.columns:
    features.append('Open')

print(f"We'll use these features: {features}")

X = df[features]
y = df[target]

print("\n✂️ STEP 5: Splitting data for training and testing...")

split_point = int(0.8 * len(df))

X_train = X[:split_point]
X_test = X[split_point:]
y_train = y[:split_point]
y_test = y[split_point:]

print(f"Training data: {len(X_train)} days")
print(f"Testing data: {len(X_test)} days")

print("\n🤖 STEP 6: Training the Linear Regression model...")

model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Model training complete!")

importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.coef_
})
print("\nFeature importance (coefficients):")
print(importance.sort_values('Importance', ascending=False))

print("\n🔮 STEP 7: Making predictions...")

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

print("✅ Predictions made!")

print("\n📏 STEP 8: Checking model accuracy...")

train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

print("\n📊 MODEL PERFORMANCE:")
print("-" * 30)
print(f"Training Data:")
print(f"  • RMSE (error in $): ${train_rmse:.2f}")
print(f"  • R² Score: {train_r2:.3f}")
print(f"\nTesting Data:")
print(f"  • RMSE (error in $): ${test_rmse:.2f}")
print(f"  • R² Score: {test_r2:.3f}")

if test_r2 > 0.8:
    print("✨ Great! The model explains over 80% of price movements!")
elif test_r2 > 0.6:
    print("👍 Good! The model explains over 60% of price movements.")
else:
    print("🤔 The model is basic, but that's okay for learning!")

print("\n🎨 STEP 9: Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Stock Price Prediction Results', fontsize=16)

axes[0, 0].plot(y_train.values, label='Actual Price', color='blue', alpha=0.7)
axes[0, 0].plot(train_predictions, label='Predicted Price', color='red', alpha=0.7)
axes[0, 0].set_title('Training Data: Actual vs Predicted')
axes[0, 0].set_xlabel('Time (days)')
axes[0, 0].set_ylabel('Stock Price ($)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(y_test.values, label='Actual Price', color='blue', linewidth=2)
axes[0, 1].plot(test_predictions, label='Predicted Price', color='red', linewidth=2)
axes[0, 1].set_title('Test Data: Actual vs Predicted')
axes[0, 1].set_xlabel('Time (days)')
axes[0, 1].set_ylabel('Stock Price ($)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(y_test, test_predictions, alpha=0.6)
axes[1, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'red', linestyle='--', linewidth=2)
axes[1, 0].set_title('Prediction Accuracy (Test Data)')
axes[1, 0].set_xlabel('Actual Price ($)')
axes[1, 0].set_ylabel('Predicted Price ($)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].barh(importance['Feature'], importance['Importance'])
axes[1, 1].set_title('Feature Importance')
axes[1, 1].set_xlabel('Coefficient Value')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n🔮 STEP 10: Let's predict tomorrow's price!")

latest_data = X.iloc[-1:].copy()
tomorrow_prediction = model.predict(latest_data)[0]
latest_actual = y.iloc[-1]

print(f"\n📅 Today's closing price: ${latest_actual:.2f}")
print(f"📅 Tomorrow's predicted price: ${tomorrow_prediction:.2f}")

price_change = tomorrow_prediction - latest_actual
if price_change > 0:
    print(f"📈 Predicted increase: +${price_change:.2f}")
elif price_change < 0:
    print(f"📉 Predicted decrease: -${abs(price_change):.2f}")
else:
    print("➡️ Predicted no change")

print("\n" + "=" * 60)
print("📝 PROJECT SUMMARY")
print("=" * 60)
print("✓ Loaded and prepared stock data")
print("✓ Created useful features for prediction")
print("✓ Trained a Linear Regression model")
print("✓ Tested the model on unseen data")
print(f"✓ Model accuracy (R²): {test_r2:.3f}")
print(f"✓ Average prediction error: ${test_rmse:.2f}")
print("\n🎉 Congratulations! You've built your first stock prediction model!")
print("Remember: This is for learning - real stock prediction is much more complex!")
print("=" * 60)

save_option = input("\n💾 Would you like to save predictions to CSV? (yes/no): ")
if save_option.lower() == 'yes':
    results_df = pd.DataFrame({
        'Date': df['Date'].iloc[split_point:].values,
        'Actual_Price': y_test.values,
        'Predicted_Price': test_predictions,
        'Error': y_test.values - test_predictions
    })
    results_df.to_csv('stock_predictions.csv', index=False)
    print("✅ Predictions saved to 'stock_predictions.csv'")
