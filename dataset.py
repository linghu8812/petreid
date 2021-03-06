import os
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import torch
import numpy as np


def build_transform(img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


def read_image(path, name):
    src_img = Image.open(os.path.join(path, name))
    img = np.zeros((max(src_img.size), max(src_img.size), 3)).astype(np.uint8)
    img[:src_img.size[1], :src_img.size[0], :] = np.array(src_img)
    src_img = Image.fromarray(img)
    return src_img


class TestData(Dataset):
    def __init__(self, data_path, csv_file, transform, bad_file=None):
        self.data_path = data_path
        self.dataset = pd.read_csv(csv_file).values
        self.transform = transform

        self.bad_data_list = {}
        if bad_file is not None:
            with open(bad_file, 'r') as f:
                data = f.readlines()
            for line in data:
                name, class_id = line.strip().split(' ')
                self.bad_data_list[name] = int(class_id)

    def __getitem__(self, item):
        image_name1, image_name2 = self.dataset[item][0], self.dataset[item][1]
        src_img1 = read_image(self.data_path, self.dataset[item][0])
        if image_name1 in self.bad_data_list:
            rotate_time = self.bad_data_list[image_name1]
            src_img1 = Image.fromarray(np.rot90(np.asarray(src_img1), rotate_time, axes=(1, 0)))
        src_img2 = read_image(self.data_path, self.dataset[item][1])
        if image_name2 in self.bad_data_list:
            rotate_time = self.bad_data_list[image_name2]
            src_img2 = Image.fromarray(np.rot90(np.asarray(src_img2), rotate_time, axes=(1, 0)))
        if self.transform is not None:
            tensor1 = self.transform(src_img1)
            tensor2 = self.transform(src_img2)
        else:
            tensor1 = torch.from_numpy(src_img1)
            tensor2 = torch.from_numpy(src_img2)
        return tensor1, tensor2, image_name1, image_name2

    def __len__(self):
        return len(self.dataset)
