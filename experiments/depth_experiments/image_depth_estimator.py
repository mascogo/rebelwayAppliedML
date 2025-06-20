import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from prepare_data import NYUDepthDataset, NYUDepthTransform
from model_build import DepthEstimationModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

# model = DepthEstimationModel()
# model.load_state_dict(torch.load('depth_estimation_model.pth', weights_only=True))

model.eval()

model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([384, 384]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# source = 0   # webcam
source = r"C:\Users\usuario\Pictures\FONDOS_para CHAT\DogStar-Interiors_014-sm-optimised.jpg"
# cap = cv2.VideoCapture(source)
img = cv2.imread(source)
print("img: {}".format(img))
if img.any():
    input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_image = transform(input_image).unsqueeze(0).to(device)

    # depth estimation
    with torch.no_grad():    # inference mode
        depth_map = model(input_image)

    frame_height, frame_width = img.shape[:2]
    depth_map = depth_map.squeeze().cpu().numpy()

    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_map = cv2.resize(depth_map, (frame_width, frame_height))

    cv2.imshow("Depth Map", depth_map)
    cv2.imshow("Feed", img)
    cv2.imwrite('predicted_depth.png',depth_map )

    while cv2.waitKey(1) != 27:
        pass
p = input("Press a key to close")
cv2.destroyAllWindows()