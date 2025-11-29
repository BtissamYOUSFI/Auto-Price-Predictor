import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# 1. Load the dataset
df = pd.read_csv("encoded.csv")

# 2. Select target and features
y = df["Prix"]
X = df.drop("Prix", axis=1)

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# 5. Evaluate the model
print("--- Random Forest ---")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# 6. Plot predictions vs true values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True Prix")
plt.ylabel("Predicted Prix")
plt.title("Random Forest - Predicted vs True")
plt.grid(True)
plt.show()

# 7. Export the model
joblib.dump(rf_model, "model/random_forest_model.pkl")
