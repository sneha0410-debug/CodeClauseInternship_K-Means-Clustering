import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'C:\Users\user\Desktop\Intership\customer_purchase_data.csv')

print(f"Number of samples: {len(data)}")
print(data)

features = data[['purchase_frequency', 'average_purchase_amount', 'recency']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

sse = []
for k in range(1, 7):  
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 7), sse, marker='o')  
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal k')
plt.show()

k = 3  
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

sns.scatterplot(x='purchase_frequency', y='average_purchase_amount', hue='Cluster', data=data, palette='viridis')
plt.title('Customer Segments Based on Purchase Behavior')
plt.show()

cluster_summary = data.groupby('Cluster').mean()
print(cluster_summary)
