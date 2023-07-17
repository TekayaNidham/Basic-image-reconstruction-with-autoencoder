""" usage guide
python train.py --num_epochs 10 --latent_size 256 --batch_size 128 --lr 0.0001
 """

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, random_split
from model.autoencoder import Autoencoder
import argparse
from tqdm import tqdm

# This code trains an autoencoder on the GTSRB dataset.
# The user can specify the number of epochs, latent size, batch size, and learning rate.

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs', required=False)
parser.add_argument('--latent_size', type=int, default=2, help='latent size', required=True)
parser.add_argument('--batch_size', type=int, default=128, help='batch size', required=False)
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate', required=False)
args = parser.parse_args()

# Set the hyperparameters.
num_epochs = args.num_epochs
latent_size = args.latent_size
batch_size = args.batch_size
learning_rate = args.lr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# setting up the dataset, size, input format and normalization to channels (RGB dataset)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Load the full dataset
full_dataset = torchvision.datasets.GTSRB(root='./data', transform=transform, download=True)

# Split indices for train, and test sets
dataset_size = len(full_dataset)
train_size = int(0.9 * dataset_size)
test_size = int(0.1 * dataset_size)
train_indices, test_indices = random_split(
    range(dataset_size),
    [train_size, test_size]
)

# Create subset datasets and data loaders
train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)
# saving the test set for later use (guarantee the unfamiliarity of the model)
torch.save(test_dataset, 'output/test_data.pth')
# loading the train data by batch size
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



# Model initialization
model = Autoencoder(latent_size).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
# hyperparameters learning rate and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the autoencoder
total_step = len(train_loader)
for epoch in range(num_epochs):
    with tqdm(total=total_step, unit='step') as progress_bar:
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the progress bar with batch information
            progress_bar.set_postfix({'Epoch': epoch+1, 'Loss': loss.item()})
            progress_bar.update(1)
# Save the model
torch.save(model.state_dict(), 'output/model_latent_size_' + str(latent_size) + '.pth')
print("Custom sized latent model saved successfully!")


# Model initialization with 2 sized latent space
model = Autoencoder(2).to(device)


# same previous process
total_step = len(train_loader)
for epoch in range(num_epochs):
    with tqdm(total=total_step, unit='step') as progress_bar:
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the progress bar with batch information
            progress_bar.set_postfix({'Epoch': epoch+1, 'Loss': loss.item()})
            progress_bar.update(1)
# Save the model
torch.save(model.state_dict(), 'output/model_latent_size_2.pth')
print("2 sized latent model saved successfully!")
