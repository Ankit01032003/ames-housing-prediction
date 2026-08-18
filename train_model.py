import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

# Create models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

print("="*60)
print("🏠 AMES HOUSING PRICE PREDICTION - ANN TRAINING")
print("="*60)

print("\n📊 Loading data...")
try:
    df = pd.read_csv('data/AmesHousing.csv')
    print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please make sure you have placed the CSV file in the 'data' folder.")
    exit()

# Select features
features = ['Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add',
            'Gr Liv Area', 'Bedroom AbvGr', 'Full Bath', 'Half Bath',
            'Kitchen AbvGr', 'TotRms AbvGrd', 'Lot Area', 'Total Bsmt SF',
            '1st Flr SF', '2nd Flr SF', 'Garage Area']

print(f"\n📋 Using {len(features)} features")

X = df[features].copy()
y = df['SalePrice']

print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")

# Handle missing values
print("\n🔄 Handling missing values...")
X = X.fillna(X.median())
print("✅ Missing values handled")

# Scale features
print("\n📐 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Features scaled")

# Split data
print("\n✂️ Splitting data into train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Build ANN model
print("\n🧠 Building ANN model...")
model = Sequential([
    Dense(450, activation='relu', input_shape=(X_train.shape[1],)),
    
    Dense(900, activation='relu'),
    
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("✅ Model built successfully")
model.summary()

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# Train model
print("\n🏋️ Training model...")
print("-"*60)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)
print("-"*60)

# Evaluate
print("\n📊 Evaluating model...")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📈 PERFORMANCE METRICS:")
print(f"   ✅ R² Score: {r2:.4f}")
print(f"   ✅ RMSE: ${rmse:,.2f}")
print(f"   ✅ MAE: ${mae:,.2f}")

# Calculate MAPE correctly
y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
y_pred_flat = y_pred.flatten()
mape = np.mean(np.abs((y_test_np - y_pred_flat) / y_test_np)) * 100
print(f"   ✅ MAPE: {mape:.2f}%")

# Save model and artifacts
print("\n💾 Saving model and artifacts...")
model.save('models/ann_model.h5')
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(features, f)

print("✅ Model saved to 'models/' folder")
print("   - ann_model.h5")
print("   - scaler.pkl")
print("   - feature_names.pkl")

print("\n" + "="*60)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nNext step: Run 'streamlit run app.py' to launch the web app!")