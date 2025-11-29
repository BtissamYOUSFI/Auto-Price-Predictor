import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
# 1. Load the dataset
df = pd.read_csv("normalized.csv")

# 2. Select target and features
y = df["Prix"]  # Predicting the price
X = df.drop("Prix", axis=1)

# 3. Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Support Vector Regressor": SVR()
}

# 5. Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n--- {name} ---")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # Plot predictions vs true values
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("True Prix")
    plt.ylabel("Predicted Prix")
    plt.title(f"{name} - Predicted vs True")
    plt.grid(True)
    plt.show()

    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validation R²: {scores.mean():.3f}")

