import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime



#Load the dataset
df = pd.read_csv("C:/Users/yasme/OneDrive/INST414/Motor_Vehicle_Collisions_-_Crashes.csv", low_memory=False)
df.columns = df.columns.str.lower().str.replace(' ','_')

#data cleaning - dropping rows that are missing critical info

df = df.dropna(subset = [
    'crash_date', 'crash_time', 'borough', 
    'number_of_persons_injured', 'number_of_persons_killed',
    'contributing_factor_vehicle_1'])


#convert time to hour

df['crash_time'] = pd.to_datetime(df['crash_time'], format='%H:%M', errors='coerce')
print("Bad time entries:", df['crash_time'].isna().sum())


df = df.dropna(subset=['crash_time'])
print("crash_time dtype:", df['crash_time'].dtype)


df['hour'] = df['crash_time'].dt.hour


df['contributing_factor_vehicle_1'] = df['contributing_factor_vehicle_1'].replace("Unspecified", np.nan)
df = df.dropna(subset=['contributing_factor_vehicle_1'])

#data is too big so we are
#sampling a smaller portion of the dataset for clustering 

df = df.sample(n=1000, random_state=42)


#reduce cardinality of contributing factors 

top_factors = df['contributing_factor_vehicle_1'].value_counts().nlargest(10).index
df['contributing_factor_vehicle_1'] = df['contributing_factor_vehicle_1'].apply(
    lambda x: x if x in top_factors else 'Other'
)

#The main features I am focusing on for clustering this dataset

features = df[[
    'borough','hour', 'number_of_persons_injured',
    'number_of_persons_killed', 'contributing_factor_vehicle_1'
]].copy()


#Bin injuries to reduce skew

features['injury_level'] = pd.cut(
    features['number_of_persons_injured'], 
    bins= [-1, 0, 1, 3, 10],
    labels = ['none', 'minor', 'moderate', 'major']
)

features['fatality'] = (features['number_of_persons_killed'] > 0).astype(int)
features = features.drop(['number_of_persons_injured', 'number_of_persons_killed' ], axis = 1)


categorical = ['borough', 'contributing_factor_vehicle_1', 'injury_level']
numerical = ['hour', 'fatality']


preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(drop = 'first', handle_unknown= 'ignore'), categorical),
    ('num', StandardScaler(), numerical)
])


X = preprocessor.fit_transform(features)


#Elbow Method

wcss = []
K_range = range(2,10)
for k in K_range:
    kmeans  = KMeans(n_clusters=k, random_state=42, n_init = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    

plt.figure(figsize=(8,5))
plt.plot(K_range, wcss, marker = 'o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()


#KMeans Model 

k_optimal = 4
kmeans = KMeans(n_clusters = k_optimal, random_state=42) 
clusters = kmeans.fit_predict(X)
features['cluster'] = clusters


print("\nCluster Counts:")
print(features['cluster'].value_counts())


summary = features.groupby('cluster').agg({
    'hour': 'mean', 
    'fatality': 'mean'
}).rename(columns={'hour': 'avg_hour', 'fatality': 'fatality_rate'})

print("\nCluster Summary (Mean Hour and Fatality Rate):")
print(summary)


plt.figure(figsize=(8,5))    
sns.countplot(x='cluster', data = features)
plt.title("Crash Count by Cluster")
plt.show()    

np.bincount(kmeans.labels_)