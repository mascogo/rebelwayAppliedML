import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class NYUDepthDataset(Dataset):
    def __init__(self, root_dataset, csv_file, transform=None):
        self.root_dataset = root_dataset
        self.data = pd.read_csv(os.path.join(root_dataset, csv_file))
        self.transform = transform
        self.total = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dataset, self.data.iloc[idx, 0]).replace(os.sep, '/')
        # print(f"Loading image from: {img_path}")
        depth_path = os.path.join(self.root_dataset, self.data.iloc[idx, 1]).replace(os.sep, '/')
        # print(f"Loading depth from: {depth_path}")
        image = Image.open(img_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')

        sample = {
            'image': image,
            'depth': depth,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class NYUDepthTransform:
    def __init__(self, img_size=(224, 224)):
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])


    def __call__(self, sample):
        image = self.img_transform(sample.get("image"))
        depth = self.depth_transform(sample.get("depth"))

        return {"image": image, "depth": depth}



if __name__ == "__main__":
    root_dataset = r'C:\Cursos_Rebelway\ML_for_3D_and_VFX_MAY2025\myDataSets\nyu_data'
    train_csv_path = 'data/nyu2_train.csv'
    test_csv_path = 'data/nyu2_test.csv'

    train_dataset = NYUDepthDataset(root_dataset, train_csv_path, transform=NYUDepthTransform())
    print(f"Number of training samples: {len(train_dataset)}")

    test_dataset = NYUDepthDataset(root_dataset, test_csv_path, transform=NYUDepthTransform())
    print(f"Number of test samples: {len(test_dataset)}")

    print("train_dataset: {}".format(train_dataset))
    print("train_dataset[0]: {}".format(train_dataset[0]))

    sample = train_dataset.__getitem__(0)
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}, Depth shape: {sample['depth'].shape}")


    # Example of accessing a sample
    # for i in range(len(train_dataset)):
    #     sample = train_dataset.__getitem__(i)
    #     print(f"Sample keys: {sample.keys()}")
    #     print(f"Image shape: {sample['image'].shape}, Depth shape: {sample['depth'].shape}")

