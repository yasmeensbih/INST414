import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#loading the dataset
df = pd.read_csv('synthetic_house_data.csv')

#inpsecting the data
print("Dataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())


#define features (x) and the target (y)


X = df.drop('SalePrice', axis = 1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


#normalized features to improve the models performance 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)



print(f"\nModel Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R sqaured Score: {r2:.4f}")


#plotting actual vs predicted values 

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()



#Identifying the 5 worst predictions that were made (These were the largest absolute errors)

y_pred_series = pd.Series(y_pred, index = y_test.index)
errors = np.abs(y_test - y_pred_series)
worst_indices = errors.sort_values(ascending = False).head(5).index

wrong_predictions = df.loc[worst_indices].copy()
wrong_predictions["PredictedPrice"] = y_pred_series[worst_indices]
wrong_predictions["ActualPrice"] = y_test[worst_indices]
wrong_predictions["AbsoluteError"] = errors[worst_indices]


#displaying the wrong predictions 
print("\nTop 5 Wrong Predictions:")
print(wrong_predictions)

#visualizing the data

model.fit(X_train_scaled, y_train)
importances = model.coef_
feature_names = X.columns
plt.barh(feature_names, importances)
plt.title("Feature Coefficients")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()


#Area and DistanceToCityCenter were the most influential predictors 

plt.figure(figsize=(10,6))
scatter = plt.scatter(
    y_test,
    y_pred,
    c = X_test['DistanceToCityCenter(km)'],
    cmap = 'viridis',
    alpha=0.8
)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted Sale Prices (Color = Distance to City Center)")
cbar = plt.colorbar(scatter)
cbar.set_label('Distance to City Center (km)')
plt.show()

