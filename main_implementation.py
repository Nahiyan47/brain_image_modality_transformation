
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import nibabel as nib
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset Class
class BrainTumorDataset(data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_names = sorted(os.listdir(self.root_dir))

    def __len__(self):
        return len(self.file_names) * 155  # Assuming 155 slices per volume

    def __getitem__(self, idx):
        volume_idx = idx // 155  # Which volume
        slice_idx = idx % 155    # Which slice in the volume
        
        file_path = os.path.join(self.root_dir, self.file_names[volume_idx])
        image_data = nib.load(file_path).get_fdata()

        # Load FLAIR and T2w modalities (assuming they are at index 0 and 3 respectively)
        flair_slice = torch.tensor(image_data[:, :, slice_idx, 0], dtype=torch.float32)
        t2w_slice = torch.tensor(image_data[:, :, slice_idx, 3], dtype=torch.float32)

        # Normalize the slices to range [0, 1]
        flair_slice = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min())
        t2w_slice = (t2w_slice - t2w_slice.min()) / (t2w_slice.max() - t2w_slice.min())

        return flair_slice.unsqueeze(0), t2w_slice.unsqueeze(0)  # Shape: (1, H, W)

# Initialize dataset and DataLoader
data_path = '/kaggle/input/task01braintumour/Task01_BrainTumour/imagesTr/'
dataset = BrainTumorDataset(data_path)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Define Generator and Discriminator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Define layers (example architecture)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Output channel 1

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))  # Sigmoid activation to constrain output
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Define layers (example architecture)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Output channel 1

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))  # Sigmoid activation to constrain output
        return x

# Initialize models, optimizers, and loss
G_flair_to_t2w = Generator().to(device)
G_t2w_to_flair = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(list(G_flair_to_t2w.parameters()) + list(G_t2w_to_flair.parameters()), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

# Training loop
num_epochs = 2

# Function to display generated images
def show_generated_images(flair, fake_t2w, fake_flair):
    # Convert tensors to numpy for visualization
    flair_img = flair.squeeze().cpu().detach().numpy()
    fake_t2w_img = fake_t2w.squeeze().cpu().detach().numpy()
    fake_flair_img = fake_flair.squeeze().cpu().detach().numpy()
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("FLAIR")
    plt.imshow(flair_img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Fake T2w")
    plt.imshow(fake_t2w_img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Fake FLAIR")
    plt.imshow(fake_flair_img, cmap='gray')
    plt.axis('off')
    
    plt.show()

for epoch in range(num_epochs):
    for i, (flair, t2w) in enumerate(train_loader):
        flair = flair.to(device)
        t2w = t2w.to(device)

        # Train the Generators
        optimizer_G.zero_grad()

        # Forward pass
        fake_t2w = G_flair_to_t2w(flair)
        cycle_flair = G_t2w_to_flair(fake_t2w)

        fake_flair = G_t2w_to_flair(t2w)  # Generate fake FLAIR
        cycle_t2w = G_flair_to_t2w(fake_flair)  # Generate cycle T2w

        # Clamp outputs to ensure they are between 0 and 1
        cycle_flair = torch.clamp(cycle_flair, 0, 1)
        cycle_t2w = torch.clamp(cycle_t2w, 0, 1)

        # Check for NaN values
        if torch.isnan(cycle_flair).any() or torch.isnan(cycle_t2w).any():
            print("NaN detected in cycle images!")
            continue  # Skip this iteration if NaN detected

        # Loss calculations
        loss_cycle = criterion(cycle_flair, flair) + criterion(cycle_t2w, t2w)
        loss_cycle.backward()

        optimizer_G.step()

        # Train the Discriminator
        optimizer_D.zero_grad()

        # Discriminator loss calculations (fake and real)
        real_labels = torch.ones_like(t2w).to(device)
        fake_labels_t2w = torch.zeros_like(fake_t2w).to(device)
        fake_labels_flair = torch.zeros_like(fake_flair).to(device)

        loss_D_real = criterion(D(t2w), real_labels)
        loss_D_fake_t2w = criterion(D(fake_t2w.detach()), fake_labels_t2w)
        loss_D_fake_flair = criterion(D(fake_flair.detach()), fake_labels_flair)

        loss_D = (loss_D_real + loss_D_fake_t2w + loss_D_fake_flair) / 3
        loss_D.backward()
        optimizer_D.step()

        if i % 10 == 0:  # Print every 10 steps
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Cycle Loss: {loss_cycle.item()}, D Loss: {loss_D.item()}')

            # Display generated images every 10 steps
            show_generated_images(flair, fake_t2w, fake_flair)

print("Training complete.")
