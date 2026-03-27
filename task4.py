# 🔹 Import required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 🔹 Step 1: Load Dataset
df = pd.read_csv("housing.csv")
print(df.head())

# 🔹 Step 2: Explore Data
print(df.info())
print(df.isnull().sum())
print(df.describe())

# 🔹 Step 3: Handle Missing Values (Corrected)
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())

# 🔹 Step 4: Feature Engineering
df['AveRooms'] = df['total_rooms'] / df['households']
df['AveBedrms'] = df['total_bedrooms'] / df['households']
df['AveOccup'] = df['population'] / df['households']

# 🔹 Step 5: Rename Columns
df.rename(columns={
    'median_income': 'MedInc',
    'housing_median_age': 'HouseAge',
    'population': 'Population',
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'median_house_value': 'MedHouseVal'
}, inplace=True)

# 🔍 Check columns
print("\nColumns after processing:\n", df.columns)

# 🔹 Step 6: Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 🔹 Step 7: Select Features & Target
X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
        'Population', 'AveOccup', 'Latitude', 'Longitude']]
y = df['MedHouseVal']

# 🔹 Step 8: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Step 9: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Step 10: Build Ridge Regression Model
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# 🔹 Step 11: Predictions
y_pred = model.predict(X_test_scaled)

# 🔹 Step 12: Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# 🔹 Step 13: Coefficient Interpretation
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nFeature Importance:\n", coeff_df)

# 🔹 Step 14: Plot Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")

# Perfect prediction line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2)

plt.show()
