import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from prepare_data import NYUDepthDataset, NYUDepthTransform	
from model_build import DepthEstimationModel
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: '{device}'")

model = DepthEstimationModel()
model.load_state_dict(torch.load('depth_estimation_model.pth', weights_only=True))
model.to(device)

def predict_depth(model, image_path, transform, output_path="predicted_depth.png"):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform.img_transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        predicted_depth = model(image_tensor)
        predicted_depth = predicted_depth.squeeze().cpu().numpy()

    plt.imsave(output_path, predicted_depth, cmap='gray')
    print(f"Predicted depth saved to {output_path}")    


if __name__ == "__main__":
    # image_path = 'death-star-battle-trench1.jpg'  # Change to your test image path
    image_path = r'C:\Cursos_Rebelway\ML_for_3D_and_VFX_MAY2025\myDataSets\nyu_data\data\nyu2_test\00087_colors.png'  # Change to your test image path
    predict_depth(model, image_path, NYUDepthTransform(), output_path="predicted_depth.png")