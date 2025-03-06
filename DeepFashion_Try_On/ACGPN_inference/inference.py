import os
import torch
import numpy as np
import cv2
from torch.autograd import Variable
from options.train_options import TrainOptions
from models.models import create_model
import util.util as util
from data.data_loader import CreateDataLoader

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
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    for i, dress in enumerate(dresses):
        # Load dress mask
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

        # Save the output
        output_image = (fake_image[0].permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        output_image = (output_image * 255).astype(np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"output/dress_{i}.png", output_image)
        print(f"Saved output: output/dress_{i}.png")

# Load single person image and its related data
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# TODO GET THE IMAGE FROM USER AND MAKE A RECOMMENDATION CALL

person_data = next(iter(dataset))  # Fetch the first person sample


os.makedirs("input_data", exist_ok=True)


person_image = person_data['image']
person_label = person_data['label']
person_mask = person_data['mask']
person_pose = person_data['pose']

cv2.imwrite(f"output/person_image.png", person_image)

# Debugging prints
# print("Person Image Shape:", person_image.shape)
# print("Label Shape:", person_label.shape)
# print("Mask Shape:", person_mask.shape)
# print("Pose Shape:", person_pose.shape)

# Apply all dresses in the dataset to the single person image
apply_dresses(person_image, person_label, person_mask, person_pose, dataset)
