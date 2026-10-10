import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


df: pd.DataFrame = pd.read_csv(Path("data/Walmart.csv"))
Y: pd.DataFrame = df["Weekly_Sales"]
X: pd.DataFrame = df[
    ["Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
].to_numpy()

scaler = StandardScaler()
X = scaler.fit_transform(X)

m, n = X.shape
w = np.zeros(n)
b = 0.0
lr = 0.01

for i in range(10000):
    y_pred = X @ w + b
    error = y_pred - Y
    dw = (X.T @ error) / m
    db = np.sum(error) / m
    w -= lr * dw
    b -= lr * db
    
    if i%100 == 0 or i == 9999:
        cost = np.mean(error ** 2) / 2
        print(f"Epoch: {i}\nCost: {cost:.2f}")

print(w, b)

DP = np.array([[1, 36.39, 3.022, 212.9367046, 7.742]])

# Use the same scaler fitted during training
DP_scaled = scaler.transform(DP)

prediction = DP_scaled @ w + b

print(f"Predicted Weekly Sales: ${prediction[0]:,.2f}")