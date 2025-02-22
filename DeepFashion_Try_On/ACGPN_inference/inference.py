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
    arm1=torch.FloatTensor((label.cpu().numpy()==11).astype(np.int))
    arm2=torch.FloatTensor((label.cpu().numpy()==13).astype(np.int))
    noise=torch.FloatTensor((label.cpu().numpy()==7).astype(np.int))
    label=label*(1-arm1)+arm1*4
    label=label*(1-arm2)+arm2*4
    label=label*(1-noise)+noise*4
    return label

opt = TrainOptions().parse()

# Load model
model = create_model(opt)

def apply_dresses(person_image, dresses):
    """Applies each dress in 'dresses' to 'person_image' and saves the results."""
    for i, dress in enumerate(dresses):
        # Load dress mask
        mask_clothes = torch.FloatTensor((dress['label'].cpu().numpy() == 4).astype(np.int))
        mask_fore = torch.FloatTensor((dress['label'].cpu().numpy() > 0).astype(np.int))
        img_fore = person_image * mask_fore
        img_fore_wc = img_fore * mask_fore
        all_clothes_label = changearm(dress['label'])

        # Forward Pass
        _, fake_image, _, _, _, _, _, _, rgb, _ = model(
            Variable(dress['label'].cuda()),
            Variable(dress['edge'].cuda()),
            Variable(img_fore.cuda()),
            Variable(mask_clothes.cuda()),
            Variable(dress['color'].cuda()),
            Variable(all_clothes_label.cuda()),
            Variable(person_image.cuda()),
            Variable(dress['pose'].cuda()),
            Variable(person_image.cuda()),
            Variable(mask_fore.cuda())
        )

        # Save the output
        output_image = (fake_image[0].permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        output_image = (output_image * 255).astype(np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"output/dress_{i}.png", output_image)
        print(f"Saved output: output/dress_{i}.png")

# Load single person image
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
person_image = next(iter(dataset))['image']
print(person_image)  # Use first image as the person

# Apply all dresses in the dataset to the single person image
apply_dresses(person_image, dataset)
