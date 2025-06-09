import os

from transformers import AutoModelForImageSegmentation

# Imports
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

device = "cuda"   # "cuda"

birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
 # -- BiRefNet should be loaded with codes above, either way.
torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to(device)
birefnet.eval()
birefnet.half()

def extract_object(birefnet, imagepath):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath)
    input_images = transform_image(image).unsqueeze(0).to(device).half()

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image, mask

# Visualization
shot_path = r"./frames"
alpha_path = os.path.join(shot_path, "alpha")
os.makedirs(alpha_path, exist_ok=True)
frames = os.listdir(shot_path)
total = len(frames)
print("total frames: {}".format(total))
for i, frame in enumerate(frames):
    print("{}/{}: {}".format(i+1, total, frame))
    # img = r'C:\Cursos_Rebelway\ML_for_3D_and_VFX_MAY2025\myDataSets\ode\PRIETO03.bmp'
    img_path = os.path.join(shot_path, frame)
    if os.path.isfile(img_path):
        print("img_path: {}".format(img_path))
        new_img, mask = extract_object(birefnet, imagepath=img_path)
        new_path = os.path.join(alpha_path, "{}.{}".format(os.path.splitext(frame)[0], "png"))
        new_img.save(new_path, "PNG")
        print("\tsaved as: {}".format(os.path.basename(new_path)))
