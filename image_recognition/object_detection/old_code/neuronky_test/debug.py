# %% LOAD DEPENDECIES
import os
import numpy as np
import shelve
from pathlib import Path
import torch
import torch.utils.data
from PIL import Image
import torchvision
import pandas as pd
import re
import cv2

# import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T




# %%

class StaverDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, "scans/scans"))))
        self.masks = list(
            sorted(os.listdir(os.path.join(self.root, "ground-truth-maps/ground-truth-maps"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "scans/scans", self.imgs[idx])
        mask_path = os.path.join(self.root, "ground-truth-maps/ground-truth-maps", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        new_dim = (int(img.size[0] * 0.7), int(img.size[1] * 0.7))
        img = img.resize(size=new_dim)


        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        mask = mask.resize(size=new_dim, resample=Image.NEAREST)

        mask = 255 - np.array(mask)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        boxes = []
        pos = np.where(mask)

        inflex_point_x = list(map(int, np.where(np.diff(np.unique(pos[1])) > 2)[0]))
        inflex_point_y = list(map(int, np.where(np.diff(np.unique(pos[0])) > 2)[0]))

        if len(inflex_point_x) > 0:
            inflex_pixel_x = np.unique(pos[1])[inflex_point_x[0]]
            inflex_pixel_y = np.unique(pos[0])[inflex_point_y[0]]

            x_min_pixel = np.where(pos[1] == inflex_pixel_x)[0][-1]
            xmin = np.min(pos[1][0:x_min_pixel])
            xmax = np.max(pos[1][0:x_min_pixel])

            y_min_pixel = np.where(pos[0] == inflex_pixel_y)[0][-1]
            ymin = np.min(pos[0][0:y_min_pixel])
            ymax = np.max(pos[0][0:y_min_pixel])
            boxes.append([xmin, ymin, xmax, ymax])

            #% second box
            inflex_pixel_x_2 = np.unique(pos[1])[inflex_point_x[0]+1]
            inflex_pixel_y_2 = np.unique(pos[0])[inflex_point_y[0]+1]

            x_min_pixel_2 = np.where(pos[1] == inflex_pixel_x_2)[0][0]
            xmin = np.min(pos[1][x_min_pixel_2:])
            xmax = np.max(pos[1][x_min_pixel_2:])

            y_min_pixel_2 = np.where(pos[0] == inflex_pixel_y_2)[0][0]
            ymin = np.min(pos[0][y_min_pixel_2:])
            ymax = np.max(pos[0][y_min_pixel_2:])
            boxes.append([xmin, ymin, xmax, ymax])
        else:
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)


        # split the color-encoded mask into a set
        # of binary masks
        masks_tf = mask == obj_ids[:, None, None]
        # there is only one class
        num_objs = len(obj_ids)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks_tf = torch.as_tensor(masks_tf, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        print(img_path)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks_tf
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# %%

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# %%

path_to_dataset = "../data/staver/data/"

SCANS_DIR = "scans/scans/"
TRUTH_DIR = "ground-truth-pixel/ground-truth-pixel/"

# %%
dataset = StaverDataset(path_to_dataset, get_transform(train=False))
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn
)

a = iter(data_loader)
for idx in range(len(dataset)):
  image = next(a)
  print(image)

# %%


