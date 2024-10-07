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

# Function to load and display T2w modality (assumed to be the fourth channel)
def show_t2w_images(image_files, data_path):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # Create a 2x5 grid for displaying images
    axes = axes.ravel()  # Flatten the axes array for easy indexing

    for i, image_file in enumerate(image_files):
        # Load the .nii file
        image_path = os.path.join(data_path, image_file)
        img = nib.load(image_path).get_fdata()

        # Check the shape of the loaded image
        print(f"Image: {image_file}, Shape: {img.shape}")

        # Extract the T2w modality (assuming it's the fourth channel)
        t2w_image = img[..., 3]  # 4D data, select fourth modality (T2w)

        # Show the middle slice of the T2w modality (for visualization)
        middle_slice = t2w_image.shape[2] // 2  # Select middle slice
        axes[i].imshow(t2w_image[:, :, middle_slice], cmap='gray')
        axes[i].set_title(f"T2w - {image_file}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Call the function to show the first 10 T2w images
show_t2w_images(image_files, data_path)
