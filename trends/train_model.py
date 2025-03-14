import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

x = df.drop(columns=["target"])
y = df["target"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model = xgb.XGBRegressor(n_estimator=100, learning_rate = 0.1, random_state=42)
model.fit(x_scaled, y)

with open("xgboost_model.pkl", 'wb') as file:
    pickle.dump({"model":model, "scaler":scaler, "features":x.columns.tolist()}, file)
print('Model saved as xgboost_model.pkl')