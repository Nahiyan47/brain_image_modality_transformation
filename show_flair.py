import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the correct path to the dataset
data_path = '/kaggle/input/task01braintumour/Task01_BrainTumour/imagesTr'

# Get the list of image files
image_files = sorted([f for f in os.listdir(data_path) if f.endswith('.nii')])[:10]  # First 10 images

# Check if any image files were found
if not image_files:
    print("No .nii files found in the specified directory.")
else:
    print(f"Found {len(image_files)} image files.")

# Function to load and display FLAIR modality (assumed to be the first channel)
def show_flair_images(image_files, data_path):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # Create a 2x5 grid for displaying images
    axes = axes.ravel()  # Flatten the axes array for easy indexing

    for i, image_file in enumerate(image_files):
        # Load the .nii file
        image_path = os.path.join(data_path, image_file)
        img = nib.load(image_path).get_fdata()

        # Check the shape of the loaded image
        print(f"Image: {image_file}, Shape: {img.shape}")

        # Extract the FLAIR modality (assuming it's the first channel)
        flair_image = img[..., 0]  # 4D data, select first modality (FLAIR)

        # Show the middle slice of the FLAIR modality (for visualization)
        middle_slice = flair_image.shape[2] // 2  # Select middle slice
        axes[i].imshow(flair_image[:, :, middle_slice], cmap='gray')
        axes[i].set_title(f"FLAIR - {image_file}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Call the function to show the first 10 FLAIR images
show_flair_images(image_files, data_path)
