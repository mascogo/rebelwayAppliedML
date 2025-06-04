import os
import torch
import torch.nn as nn
from torchvision import transforms
# from torch.utils.data import DataLoader
from PIL import Image, ImageOps
    
device = "cuda" if torch.cuda.is_available() else "cpu"

labels_dict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


   
class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu = nn.Sequential(
        nn.Linear(28*28, 256),
        nn.ReLU(),       
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64,10),
    )

  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu(x)
    return logits
    

def load_model():
    model_path = os.path.join(os.environ.get("HIP"), "fashion_mnist_model.pth")      
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model
    
def fashion_run(this_node):
    model = load_model()
    image_path = this_node.parm('load_image').eval()
    image_name = os.path.basename(image_path)
    if os.path.exists(image_path):
        image = Image.open(image_path).convert('L').resize(size=(28,28))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((28, 28))
        ])
        image = transform(image).to(device)
        with torch.no_grad():
            output = model(image)
            print(output)
            prediction = torch.argmax(output).item()
            hou.ui.displayMessage(f'Image {image_name} is a : {labels_dict.get(prediction)}.', buttons=('ok',))
            
    else:
        hou.ui.displayMessage('Path: {} not found.'.format(image_path), buttons=('ok',))
        
def button_run():
    name = hou.node(".").parms()[0].eval()
    parent = hou.node(".").parent()
    input = hou.node(".").input(0)
    this_node = hou.node('.')
    fashion_run(this_node)