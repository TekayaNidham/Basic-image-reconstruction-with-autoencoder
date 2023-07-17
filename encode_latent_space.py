import torch
import numpy as np
from model.autoencoder import Autoencoder
import argparse
import re

def extract_number(model_path):
    number = re.search(r"latent_size_(\d+)", model_path).group(1)
    return number

def encode_images(model_path, test_loader, num_samples, latent_size):
    model = Autoencoder(latent_size).to(device)
    model.load_state_dict(torch.load(model_path))
    latent_space = []
    count = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            encoded = model.encoder(images)
            #flattens the encoded tensor
            encoded = encoded.view(encoded.size(0), -1)
            #converts the encoded tensor to a NumPy array on the CPU detatches it from the computation graph
            latent_space.append(encoded.detach().cpu().numpy())
            count += len(images)
            if count >= num_samples:
                break
    
    latent_space = np.concatenate(latent_space, axis=0)
    return latent_space


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='model.pth', help='path for saving the model')
parser.add_argument('--test_data', type=str, default='test_data.pth', help='path for test data')
parser.add_argument('--num_samples', type=int, default=1000, help='number of samples to be used for latent space visualization')
args = parser.parse_args()

model_path = args.model_path
test_data = args.test_data
num_samples = args.num_samples

model_path2 = 'output/model_latent_size_2.pth'
test_dataset = torch.load(test_data)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latent_size = int(extract_number(model_path))


latent_space = encode_images(model_path, test_loader, num_samples, latent_size)
latent_space_light = encode_images(model_path2, test_loader, num_samples, latent_size=2)

# Save the latent space representations
np.save('output/latent_space_size_' + str(latent_size) + '.npy', latent_space)
np.save('output/latent_space_size_2.npy', latent_space_light)
print("Latent spaces representations saved successfully!")