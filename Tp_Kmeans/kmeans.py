from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

########## Partie 1 ##########

# 1. Charger le dataset
df = pd.read_csv('Mall_Customers.csv')
print("Aperçu du dataset :")
print(df.head())
print("\nInformations générales :")
print(df.info())     #infos generales
print("\nValeurs manquantes :")
print(df.isnull().sum())  #valeurs manquantes


df_selected = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# 3. Standardisation
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

print("\nDonnées standardisées :")
print(df_scaled[:5])

########## Partie 2 ##########

plt.figure(figsize=(7,5))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Age'])
plt.xlabel("Revenu annuel (k$)")
plt.ylabel("Spending Score")
plt.title("Scatter plot 3 variables (Age en couleur)")
plt.colorbar(label="Âge")
plt.show()

df_selected.hist() #pour fair un histograme comment les valeurs se répartissent
plt.suptitle("Distribution des variables")
plt.show()

sns.pairplot(df_selected)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df_selected.corr(), annot=True, cmap="coolwarm")
plt.title("Matrice des corrélations")
plt.show()

########## Partie 3 ##########

## Test de plusieurs valeurs de k
inertia = []
silhouette_scores = []

for k in range(2, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df_scaled)

    inertia.append(model.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, model.labels_))

#Méthode du coude
plt.figure(figsize=(8,5))
plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertia")
plt.title("Méthode du coude")
plt.grid(True)
plt.show()

#Score de silhouette
plt.figure(figsize=(8,5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Score de silhouette")
plt.title("Score silhouette pour différents k")
plt.grid(True)
plt.show()

#K=5
k_optimal = 6
model_final = KMeans(n_clusters=k_optimal, random_state=42)
model_final.fit(df_scaled)

clusters = model_final.labels_

#ajouter une cologne cluster
df['Cluster'] = clusters
print(df.head())

########## Partie 4 ##########
cluster_profiles = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print(cluster_profiles)

#clusters 2D
centers = model_final.cluster_centers_
centers_original = scaler.inverse_transform(centers)

plt.figure(figsize=(8,6))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='tab10')
plt.scatter(centers_original[:, 1], centers_original[:, 2], s=200, c='black', marker='X')
plt.title("Clusters et centroïdes")
plt.xlabel("Revenu annuel (k$)")
plt.ylabel("Spending Score")
plt.show()

#clusters 3D

from mpl_toolkits.mplot3d import Axes3D

# Scatter 3D des points
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'],
           c=df['Cluster'], cmap='tab10')

# Calcul des centroïdes remis à l'échelle originale
centers = model_final.cluster_centers_
centers_original = scaler.inverse_transform(centers)

# Scatter 3D des centroïdes
ax.scatter(centers_original[:, 0], centers_original[:, 1], centers_original[:, 2],c='black', s=300, marker='X', label='Centroïdes')

ax.set_xlabel('Âge')
ax.set_ylabel('Revenu annuel (k$)')
ax.set_zlabel('Spending Score')
plt.title("Clusters en 3D avec centroïdes")
ax.legend()

plt.show()


