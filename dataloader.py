import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.transforms import transforms, InterpolationMode
import glob
import torch.nn.functional as F


class CustomDataset(Dataset):
    def __init__(self, voc_root, kge, txt_name: str = "train.txt", transform=None):
        self.kge = kge
        self.root = os.path.join(voc_root, "VOC_landslide")
        txt_path = os.path.join(self.root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            self.file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.mask_dir = os.path.join(self.root, 'SegmentationObject')
        self.knowledge_dir = os.path.join(self.root, 'Knowledge')
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def onehot_encoder(self, mask, num_classes=2):
        onehot = torch.zeros(num_classes, mask.shape[1], mask.shape[2])
        onehot[0] = (mask == 0) * 1
        onehot[1] = (mask != 0) * 1
        return onehot

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_name = os.path.join(self.image_dir, file_name + ".jpg")
        mask_name = os.path.join(self.mask_dir, file_name + ".png")

        image = Image.open(img_name)
        mask = Image.open(mask_name)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # 执行 one-hot 编码
        mask_onehot = self.onehot_encoder(mask)

        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),  # 使用最近邻插值
        ])
        knowledge_value = torch.tensor(0)
        if self.kge:
            knowledge_list = os.listdir(self.knowledge_dir)
            tmp = []
            for knowledge in knowledge_list:
                path = os.path.join(os.path.join(self.root, 'Knowledge'), knowledge, f"{file_name}.*")
                matched_files = glob.glob(path)
                k_image = Image.open(matched_files[0])
                k_tensor = torch.from_numpy(np.array(transform(k_image)))
                tmp.append(k_tensor)
            knowledge_value = torch.stack(tmp, dim=0)
        return image, mask_onehot, knowledge_value.div(255)
