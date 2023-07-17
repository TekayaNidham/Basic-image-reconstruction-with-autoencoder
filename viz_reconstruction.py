import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
from model.autoencoder import Autoencoder
import argparse
import re

def extract_number(model_path):
    number = re.search(r"latent_size_(\d+)", model_path).group(1)
    return number


parser = argparse.ArgumentParser()
parser.add_argument('--test_data', type=str, default='output/test_data.pth', help='path for test data', required=False)
parser.add_argument('--model_path', type=str, default='output/model_latent_size_256.pth', help='path for saving the model', required=True)   
args = parser.parse_args()

test_data = args.test_data
model_path = args.model_path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path2 = 'output/model_latent_size_2.pth'


def viz_reconstructions(model_path, model_path2, test_data):
    """ model2 is the model with bottleneck dim 2 """



    # Load the trained models
    model1 = Autoencoder(int(extract_number(model_path))).to(device)
    model1.load_state_dict(torch.load(model_path))
    model1.eval()

    model2 = Autoencoder(2).to(device)
    model2.load_state_dict(torch.load(model_path2))
    model2.eval()

    # Load the test dataset
    test_dataset = torch.load(test_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Get a batch of test images
    images, _ = next(iter(test_loader))
    images = images.to(device)

    # Reconstruct the images using model1
    with torch.no_grad():
        reconstructed1 = model1(images)

    # Reconstruct the images using model2
    with torch.no_grad():
        reconstructed2 = model2(images)

    # Convert the images to CPU and denormalize
    images = images.cpu().numpy() * 0.5 + 0.5
    reconstructed1 = reconstructed1.cpu().numpy() * 0.5 + 0.5
    reconstructed2 = reconstructed2.cpu().numpy() * 0.5 + 0.5

    # Display a batch of images and their reconstructions
    fig, axes = plt.subplots(8, 3, figsize=(10, 20))

    for i in range(8):
        # Original images
        axes[i, 0].imshow(np.transpose(images[i], (1, 2, 0)))
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Original')

        # Reconstructed images from model1
        axes[i, 1].imshow(np.transpose(reconstructed1[i], (1, 2, 0)))
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Reconstructed with custom model')

        # Reconstructed images from model2
        axes[i, 2].imshow(np.transpose(reconstructed2[i], (1, 2, 0)))
        axes[i, 2].axis('off')
        axes[i, 2].set_title('Reconstructed bottlenck dim 2')

    plt.tight_layout()
    plt.show()


viz_reconstructions(model_path, model_path2, test_data)