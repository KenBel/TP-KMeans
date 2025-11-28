import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Étape 1 : charger le dataset
df = pd.read_csv('Mall_Customers.csv')

# Étape 2 : sélectionner les variables
df_selected = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Étape 3 : standardisation
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Étape 4 : DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
db_labels = dbscan.fit_predict(df_scaled)

# Ajouter les labels au dataset
df['Cluster_DBSCAN'] = db_labels
print(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster_DBSCAN']].head())

# Visualisation 2D
plt.figure(figsize=(8,6))
plt.scatter(df['Annual Income (k$)'],
            df['Spending Score (1-100)'],
            c=db_labels, cmap='tab10')
plt.xlabel("Revenu annuel (k$)")
plt.ylabel("Spending Score")
plt.title("Clustering avec DBSCAN")
plt.show()
