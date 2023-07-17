from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import argparse

args = argparse.ArgumentParser()
args.add_argument('--latent_space', type=str, default='output/latent_space_size_256.npy', help='path for latent space')
args.add_argument('--num_clusters', type=int, default=43, help='number of clusters')
args = args.parse_args()

latent_space = args.latent_space
n_clusters = args.clusters

latent_space = np.load(latent_space)

def visualize_clusters(latent_space, n_clusters):
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(latent_space)

    # Visualize the clusters in the latent space
    plt.scatter(latent_space[:, 0], latent_space[:, 1], c=cluster_labels, cmap='viridis')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Latent Space Clusters')
    plt.colorbar()
    plt.show()

scaler = StandardScaler()
latent_space_scaled = scaler.fit_transform(latent_space.reshape(latent_space.shape[0], -1))


            # Reduce dimensionality using PCA before applying UMAP
pca = PCA(n_components=50)
latent_space_pca = pca.fit_transform(latent_space_scaled)

            # Perform dimensionality reduction using UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=50)
embedding = reducer.fit_transform(latent_space_pca)    

visualize_clusters(embedding, n_clusters)
