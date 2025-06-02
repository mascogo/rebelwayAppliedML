from sys import prefix

from torchvision import datasets, transforms
from PIL import Image
import torch
import mnist

device = "cuda"

model = mnist.Net().to(device)
model.load_state_dict(torch.load("mnist_base_model.pth"))
model.eval()

image_path = "numi.png"

image = Image.open(image_path).convert("L")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((28, 28)),
])

image = transform(image).to(device)

with torch.inference_mode():
    output = model(image)   # .unsqueeze(0))
    prediction = torch.argmax(output).item()

print(f"Prediction: {prediction}")