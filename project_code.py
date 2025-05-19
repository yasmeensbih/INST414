import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 

from sklearn.ensemble import RandomForestRegressor


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


#performing linear regression 

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)



print(f"\nModel Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R Squared Score: {r2:.4f}")


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



##Extending my code starts here 

#add a set number of clusters

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
X_clustered = kmeans.fit_predict(X_train_scaled)



#Add the cluster labels to the training set
X_train_clustered = pd.DataFrame(X_train_scaled, columns = X.columns)
X_train_clustered['Cluster'] = X_clustered

#Apply clustering to test set too
X_test_clustered = pd.DataFrame(X_test_scaled, columns= X.columns)
X_test_clustered['Cluster'] = kmeans.predict(X_test_scaled)


#Resetting the y indices
y_train_clustered = y_train.reset_index(drop=True)
y_test_clustered = y_test.reset_index(drop=True)


#Training the seperate models per cluster 

cluster_results = []

for cluster_id in range(k):
    #filter by cluster
    X_train_c = X_train_clustered[X_train_clustered['Cluster'] == cluster_id].drop('Cluster', axis=1)
    y_train_c = y_train_clustered[X_train_clustered['Cluster'] == cluster_id]
    
    X_test_c = X_test_clustered[X_test_clustered['Cluster'] == cluster_id].drop('Cluster', axis=1)
    y_test_c = y_test_clustered[X_test_clustered['Cluster'] == cluster_id]
    
    
    #Train the model
    model_c = LinearRegression()
    model_c.fit(X_train_c, y_train_c)
    y_pred_c = model_c.predict(X_test_c)
    
    
    #Evaluate the results
    
    mse_c = mean_squared_error(y_test_c, y_pred_c)
    rmse_c = np.sqrt(mse_c)
    r2_c = r2_score(y_test_c, y_pred_c)
    
    cluster_results.append({
        "Cluster": cluster_id,
        "Num Test Samples": len(y_test_c),
        "R2 Score": r2_c,
        "RMSE": rmse_c
    })
    
    
print("\nCluster Model Results:")
for result in cluster_results:
    print(f"Cluster {result['Cluster']} - Samples: {result['Num Test Samples']} | RMSE: {result['RMSE']:.2f} | R2: {result['R2 Score']:.4f}")
    
    

plt.figure(figsize=(8,6))
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c= X_clustered, cmap='viridis', alpha=0.6)
plt.title("KMeans Clustering of Training Data")
plt.xlabel("Area (scaled)")
plt.ylabel("Bedrooms (scaled)")
plt.colorbar(label = "Cluster")
plt.grid(True)
plt.show()



#training random forest on the same scaled features 

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)


#evaluating the Random Forest Model 

rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)



print("\nRandom Forest Model Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rf_rmse:.2f}")
print(f"R-Squared Score: {rf_r2:.4f}")


#random forest feature importances

rf_importances = rf_model.feature_importances_


plt.figure(figsize=(6,4))
plt.barh(X.columns, rf_importances)
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.grid(True)
plt.tight_layout()
plt.show()