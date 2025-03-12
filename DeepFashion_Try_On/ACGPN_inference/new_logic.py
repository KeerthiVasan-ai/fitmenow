import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from options.train_options import TrainOptions
from models.models import create_model
from data.data_loader import CreateDataLoader
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Initialize options and model
opt = TrainOptions().parse()
model = create_model(opt)

# Function to improve edge detection for dress segmentation
def get_edge_mask(image_path):
    """Enhance edge detection for better segmentation."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img, 50, 150)
    return edges

# Function to blend dress smoothly
def blend_images(foreground, background, mask):
    """Perform Poisson blending for seamless results."""
    center = (background.shape[1] // 2, background.shape[0] // 2)
    blended = cv2.seamlessClone(foreground, background, mask, center, cv2.NORMAL_CLONE)
    return blended

# Function to apply dress fitting
def apply_dress_fitting(person_data, dress_path):
    """Applies a dress to a person image with improved alignment and blending."""
    
    person_image = person_data["image"]
    label = person_data["label"]
    mask = person_data["mask"]
    pose = person_data["pose"]
    
    # Load dress image
    dress = cv2.imread(dress_path)
    dress = cv2.resize(dress, (192, 256))  # Resize to match person image size
    
    # Generate dress edge map
    edge_mask = get_edge_mask(dress_path)

    # Apply Model
    _, fake_image, _, _, _, _, _, _, rgb, _ = model(
        Variable(label.cuda()),
        Variable(torch.tensor(edge_mask).cuda()),
        Variable(person_image.cuda()),
        Variable(mask.cuda()),
        Variable(torch.tensor(dress).cuda()),
        Variable(label.cuda()),
        Variable(person_image.cuda()),
        Variable(pose.cuda()),
        Variable(person_image.cuda()),
        Variable(mask.cuda())
    )

    # Convert model output to image format
    output_image = (fake_image[0].permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
    output_image = (output_image * 255).astype(np.uint8)

    # Blend with original image for better natural look
    mask_foreground = (mask[0].cpu().numpy() * 255).astype(np.uint8)
    output_image = blend_images(output_image, person_image.cpu().numpy(), mask_foreground)

    # Save the output
    output_path = "output/fitted_result.png"
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"Saved improved result: {output_path}")

    # Display
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Load dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# Find person data
person_data = next(iter(dataset))
# for data in dataset:
#     if data["name"] == ["002103_0.jpg"]:  # Change this to your target person's image
#         person_data = data
#         break

# if person_data is None:
#     raise ValueError("Person image not found in dataset!")

dress_images = ['/kaggle/input/viton-dataset/ACGPN_TestData/test_color/003069_1.jpg', 
                '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/013245_1.jpg', 
                '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/005101_1.jpg', 
                '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/004904_1.jpg', 
                '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/010984_1.jpg'
            ]

# Apply each dress to the person image
os.makedirs("output", exist_ok=True)  # Ensure output folder exists

for dress_path in dress_images:
    apply_dress_fitting(person_data, dress_path)
