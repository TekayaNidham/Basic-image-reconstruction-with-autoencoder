import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--latent_space', type=str, default='output/latent_space_size_256.npy', help='path for latent space')
parser.add_argument('--vis_type', type=str, default='umap', help='visualization type')
args = parser.parse_args()

latent_space = args.latent_space
vis_type = args.vis_type


def tsne_show(latent_space1, latent_space2, n_components1=50, n_components2=8):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = ['Latent Space Custom Size', 'Latent Space Dim 2']

    for i, latent_space in enumerate([latent_space1, latent_space2]):
        ax = axes[i]

        if i == 0:
            # Scale the latent space representations
            scaler = StandardScaler()
            latent_space_scaled = scaler.fit_transform(latent_space.reshape(latent_space.shape[0], -1))


            # Reduce dimensionality using PCA before applying t-SNE
            pca = PCA(n_components=50)
            latent_space_pca = pca.fit_transform(latent_space_scaled)

            # Perform dimensionality reduction using t-SNE
            reducer = TSNE(perplexity=30, n_iter=5000, random_state=42)
            embedding = reducer.fit_transform(latent_space_pca)

            # Visualize the latent space embeddings with t-SNE
            ax.scatter(embedding[:, 0], embedding[:, 1], s=5)
        else:

            # Visualize the latent space embeddings with a scatter plot
            ax.scatter(latent_space[:, 0], latent_space[:, 1], s=5)

        ax.set_title(titles[i])
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')

    plt.tight_layout()
    plt.show()




def umap_show(latent_space1, latent_space2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = ['Latent Space Custom Size', 'Latent Space Dim 2']

    for i, latent_space in enumerate([latent_space1, latent_space2]):
        ax = axes[i]

        if i == 0:
            # Scale the latent space representations
            scaler = StandardScaler()
            latent_space_scaled = scaler.fit_transform(latent_space.reshape(latent_space.shape[0], -1))


            # Reduce dimensionality using PCA before applying UMAP
            pca = PCA(n_components=50)
            latent_space_pca = pca.fit_transform(latent_space_scaled)

            # Perform dimensionality reduction using UMAP
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=50)
            embedding = reducer.fit_transform(latent_space_pca)

            # Visualize the latent space embeddings with UMAP
            ax.scatter(embedding[:, 0], embedding[:, 1], s=5)
        else:


            # Visualize the latent space embeddings with a scatter plot
            ax.scatter(latent_space[:, 0], latent_space[:, 1], s=5)

        ax.set_title(titles[i])
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')

    plt.tight_layout()
    plt.show()




# Load the latent space representations
latent_space_data = np.load(latent_space)

latent_space_light = np.load('output/latent_space_size_2.npy')

if vis_type == 'umap':
    umap_show(latent_space_data, latent_space_light)
elif vis_type == 'tsne':
    tsne_show(latent_space_data, latent_space_light)
else:
    print('Invalid visualization type, only support umap or tsne')




