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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    dress_tensor = transform(dress_img).unsqueeze(0)  # Add batch dimension

    return dress_tensor, dress_img

# Function to compute edge map
def compute_edge_map(dress_img):
    gray = cv2.cvtColor(dress_img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 50, 150)  # Apply Canny edge detection

    # Convert edges to tensor and normalize
    edge_tensor = TF.to_tensor(edges).unsqueeze(0)  # Shape: (1, 1, H, W)
    return edge_tensor

def changearm(label):
    arm1 = torch.FloatTensor((label.cpu().numpy() == 11).astype(np.int_))
    arm2 = torch.FloatTensor((label.cpu().numpy() == 13).astype(np.int_))
    noise = torch.FloatTensor((label.cpu().numpy() == 7).astype(np.int_))
    label = label * (1 - arm1) + arm1 * 4
    label = label * (1 - arm2) + arm2 * 4
    label = label * (1 - noise) + noise * 4
    return label

# Function to apply a dress onto a person
def apply_dress_to_user(person_data, dress_path):
    """Loads a dress, computes the edge map, and applies it to the person."""

    # Extract person data
    person_image = person_data['image'].cuda()
    person_label = person_data['label'].cuda()
    person_mask = person_data['mask'].cuda()
    person_pose = person_data['pose'].cuda()

    all_clothes_label = changearm(person_label)

    # Load dress image and compute edge map
    dress_tensor, dress_np = load_dress_image(dress_path)
    edge_tensor = compute_edge_map(dress_np)

    dress_tensor, edge_tensor = dress_tensor.cuda(), edge_tensor.cuda()

    # Forward pass through the model
    with torch.no_grad():  # Disable gradient computation for efficiency
        _, fake_image, _, _, _, _, _, _, _, _ = model(
            Variable(person_label),
            Variable(edge_tensor),
            Variable(person_image),
            Variable(person_mask),
            Variable(dress_tensor),
            Variable(all_clothes_label.cuda()),
            Variable(person_image),
            Variable(person_pose),
            Variable(person_image),
            Variable(person_mask)
        )

    # Convert the output to an image
    fake_image = fake_image[0].cpu().permute(1, 2, 0).detach().numpy()
    fake_image = (fake_image + 1) / 2  # Normalize to [0, 1]
    fake_image = (fake_image * 255).astype(np.uint8)

    # Blend the output with the original image
    blended_image = cv2.addWeighted(dress_np, 0.4, fake_image, 0.6, 0)

    # Save the output
    output_path = f"output/fitted_{os.path.basename(dress_path)}"
    cv2.imwrite(output_path, cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))
    print(f"Saved fitted dress: {output_path}")

# Load dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# Find person data
person_data = next(iter(dataset))

# List of dress images from folder
dress_images = [
    '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/003069_1.jpg',
    '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/013245_1.jpg',
    '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/005101_1.jpg',
    '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/004904_1.jpg',
    '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/010984_1.jpg'
]

# Apply each dress to the person image
os.makedirs("output", exist_ok=True)  # Ensure output folder exists

for dress_path in dress_images:
    apply_dress_to_user(person_data, dress_path)
