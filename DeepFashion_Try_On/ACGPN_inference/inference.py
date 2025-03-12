import os
import torch
import numpy as np
import cv2
from torch.autograd import Variable
from options.train_options import TrainOptions
from models.models import create_model
import util.util as util
from data.data_loader import CreateDataLoader
import warnings
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

def changearm(label):
    arm1 = torch.FloatTensor((label.cpu().numpy() == 11).astype(np.int_))
    arm2 = torch.FloatTensor((label.cpu().numpy() == 13).astype(np.int_))
    noise = torch.FloatTensor((label.cpu().numpy() == 7).astype(np.int_))
    label = label * (1 - arm1) + arm1 * 4
    label = label * (1 - arm2) + arm2 * 4
    label = label * (1 - noise) + noise * 4
    return label

# Parse options
opt = TrainOptions().parse()

# Load model
model = create_model(opt)

def apply_dresses(person_image, label, mask, pose, dresses):
    """Applies each dress in 'dresses' to 'person_image' and saves the results."""
    
    os.makedirs("output", exist_ok=True)

    for i, dress in enumerate(dresses):

        print(dress["name"])

        mask_clothes = torch.FloatTensor((label.cpu().numpy() == 4).astype(np.int_))
        mask_fore = torch.FloatTensor((label.cpu().numpy() > 0).astype(np.int_))
        img_fore = person_image * mask_fore
        img_fore_wc = img_fore * mask_fore
        all_clothes_label = changearm(label)

        # Forward Pass
        _, fake_image, _, _, _, _, _, _, rgb, _ = model(
            Variable(label.cuda()),
            Variable(dress['edge'].cuda()),
            Variable(img_fore.cuda()),
            Variable(mask_clothes.cuda()),
            Variable(dress['color'].cuda()),
            Variable(all_clothes_label.cuda()),
            Variable(person_image.cuda()),
            Variable(pose.cuda()),
            Variable(person_image.cuda()),
            Variable(mask_fore.cuda())
        )

        # Save output
        output_image = (fake_image[0].permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        output_image = (output_image * 255).astype(np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        output_path = f"output/dress_{i}.png"
        cv2.imwrite(output_path, output_image)
        print(f"Saved output: {output_path}")

def find_data_by_name(dataset, target_name):
    for data in dataset:
        if data["name"] == target_name:
            return data
    return None

def find_cloth_data_by_name(dataset, cloth_data):
    return [data for data in dataset if data["name"] in cloth_data]

# Load dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# Define single person and multiple dresses
person_name = ["002103_0.jpg"]
cloth_data_list = [["011623_1.jpg"], ["013764_1.jpg"], ["005101_1.jpg"], ]
                #    ["004904_0.jpg"], ["010984_0.jpg"]]

# ['/kaggle/input/viton-dataset/ACGPN_TestData/test_color/011623_1.jpg', 
#  '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/013764_1.jpg', 
#  '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/005101_1.jpg', 
#  '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/004904_1.jpg', 
#  '/kaggle/input/viton-dataset/ACGPN_TestData/test_color/012838_1.jpg']

# Find person and clothing data
person_data = find_data_by_name(dataset, person_name)
cloths = find_cloth_data_by_name(dataset, cloth_data_list)

# Validate person data
if person_data is None:
    raise ValueError(f"Person image {person_name} not found in dataset!")

# Validate clothes data
if not cloths:
    raise ValueError("No matching clothing images found in dataset!")

# Extract person image properties
person_image = person_data['image']
person_label = person_data['label']
person_mask = person_data['mask']
person_pose = person_data['pose']

# Apply all dresses to the person image
apply_dresses(person_image, person_label, person_mask, person_pose, cloths)