import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Wholesale_customers_data.csv")

print(data.head())
print(data.info())
print(data.describe())

# padronização
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, random_state=0)
data['Cluster_KMeans'] = kmeans.fit_predict(data_scaled)

linked = linkage(data_scaled, method='ward')
num_clusters = 5
data['Cluster_Hierarchical'] = fcluster(linked, num_clusters, criterion='maxclust')

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=data['Cluster_KMeans'], palette="viridis", s=50)
plt.title("Clusters K-means")
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

plt.subplot(1, 2, 2)
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=data['Cluster_Hierarchical'], palette="plasma", s=50)
plt.title("Clusters Hierárquico")
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')


plt.tight_layout()
plt.show()
