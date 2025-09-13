import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("power consumption.csv")
df.columns = df.columns.str.strip()

# Parse DateTime
df["DateTime"] = pd.to_datetime(df["DateTime"], errors='coerce')
df = df.dropna(subset=["DateTime"])

# Time-based features
df["hour"] = df["DateTime"].dt.hour
df["day"] = df["DateTime"].dt.day
df["month"] = df["DateTime"].dt.month
df["dayofweek"] = df["DateTime"].dt.dayofweek
df["is_weekend"] = df["dayofweek"] >= 5

# Define X and y
X = df.drop(columns=["DateTime", "Zone 1", "Zone 2", "Zone 3"])
y = df[["Zone 1", "Zone 2", "Zone 3"]].sum(axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LightGBM model with tuning
params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 50],
    'max_depth': [-1, 10],
    'min_child_samples': [20, 30]
}

grid = GridSearchCV(
    LGBMRegressor(random_state=42),
    param_grid=params,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

# Evaluate
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"LightGBM Tuned MSE: {mse:.2f}")
print(f"LightGBM Tuned R² Score: {r2:.4f}")

# Save
joblib.dump(best_model, "power_consumption_lgbm.pkl")
print("Model saved as power_consumption_lgbm.pkl")
