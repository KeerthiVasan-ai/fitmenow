import os
import cv2
import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import torchvision.transforms.functional as TF
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Import your existing deep learning model setup
from options.train_options import TrainOptions
from models.models import create_model
from data.data_loader import CreateDataLoader

# Parse options
opt = TrainOptions().parse()

# Load the deep learning model
model = create_model(opt)

# Function to load and preprocess dress image
def load_dress_image(dress_path):
    dress_img = cv2.imread(dress_path)  # Read image
    dress_img = cv2.cvtColor(dress_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    dress_img = cv2.resize(dress_img, (192, 256))  # Resize to match model input

    # Normalize and convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    dress_tensor = transform(dress_img).unsqueeze(0)  # Add batch dimension

    return dress_tensor, dress_img  # Return tensor and original image

# Function to compute edge map
def compute_edge_map(dress_img):
    gray = cv2.cvtColor(dress_img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    edges = cv2.Canny(gray, 100, 200)  # Apply Canny edge detection

    # Convert edges to tensor and normalize
    edge_tensor = TF.to_tensor(edges).unsqueeze(0)  # Shape: (1, 1, H, W)
    return edge_tensor

# Function to apply a dress onto a person
def apply_dress_to_user(person_data, dress_path):
    """Loads a dress, computes the edge map, and applies it to the person."""
    
    # Extract person data
    person_image = person_data['image']
    person_label = person_data['label']
    person_mask = person_data['mask']
    person_pose = person_data['pose']

    # Load dress image and compute edge map
    dress_tensor, dress_np = load_dress_image(dress_path)
    edge_tensor = compute_edge_map(dress_np)

    # Forward pass through the model
    _, fake_image, _, _, _, _, _, _, _, _ = model(
        Variable(person_label.cuda()),
        Variable(edge_tensor.cuda()),
        Variable(person_image.cuda()),
        Variable(person_mask.cuda()),
        Variable(dress_tensor.cuda()),
        Variable(person_label.cuda()),  # Assuming same label for now
        Variable(person_image.cuda()),
        Variable(person_pose.cuda()),
        Variable(person_image.cuda()),
        Variable(person_mask.cuda())
    )

    # Convert the output to an image
    output_image = (fake_image[0].permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
    output_image = (output_image * 255).astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Save the output
    output_path = f"output/fitted_{os.path.basename(dress_path)}"
    cv2.imwrite(output_path, output_image)
    print(f"Saved fitted dress: {output_path}")

# Load dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# Find person data
person_data = None
for data in dataset:
    if data["name"] == ["002103_0.jpg"]:  # Change this to your target person's image
        person_data = data
        break

if person_data is None:
    raise ValueError("Person image not found in dataset!")

# List of dress images from folder
dress_folder = "dresses"  # Change this to your dress folder path
dress_images = ['/kaggle/input/viton-dataset/ACGPN_TestData/test_color/003069_1.jpg', 
                '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/013245_1.jpg', 
                '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/005101_1.jpg', 
                '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/004904_1.jpg', 
                '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/010984_1.jpg'
            ]

# Apply each dress to the person image
os.makedirs("output", exist_ok=True)  # Ensure output folder exists

for dress_path in dress_images:
    apply_dress_to_user(person_data, dress_path)
