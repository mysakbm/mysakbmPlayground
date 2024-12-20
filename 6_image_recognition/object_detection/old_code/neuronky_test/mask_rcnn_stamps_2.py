# %% LOAD DEPENDECIES
import os
import random

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
import matplotlib as plt
import glob
from imageio import imread
from PIL import Image

# import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
# import transforms as T
import torchvision.transforms as T
from torchvision.transforms import functional as F

# %%

path_to_dataset = "../data/staver/data/"

SCANS_DIR = "scans/scans/"
TRUTH_DIR = "ground-truth-pixel/ground-truth-pixel/"


# %%
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True,
        trainable_backbone_layers = 0 # Default byl None
        )

    for param in model.parameters():
        param.requires_grad = False

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model

# %%

def get_transform(img, mask, seed):
    # Random Horizontal Flip
    torch.manual_seed(seed)
    # img = T.RandomHorizontalFlip(p=0.5)(img)
    # img = T.RandomCrop(size=(1824, 1284))(img)
    img = T.RandomAffine(degrees=5,
                   translate=(0.1, 0.3),
                   scale=(0.85, 0.99))(img)
    torch.manual_seed(seed)
    # mask = T.RandomHorizontalFlip(p=0.5)(mask)
    # mask = T.RandomCrop(size=(1824, 1284))(mask)
    mask = T.RandomAffine(degrees=5,
                   translate=(0.1, 0.3),
                   scale=(0.85, 0.99))(mask)

    return (img, mask)

# %%

class StaverDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=False, to_tensor=False):
        self.root = root
        self.transforms = transforms
        self.to_tensor = to_tensor
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, SCANS_DIR))))
        self.masks_pixels = list(
            sorted(os.listdir(os.path.join(self.root, "ground-truth-pixel/ground-truth-pixel"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, SCANS_DIR, self.imgs[idx])
        # print(img_path)
        pixels_path = os.path.join(self.root, "ground-truth-pixel/ground-truth-pixel", self.masks_pixels[idx])
        img = Image.open(img_path).convert("RGB")

        new_dim = (int(img.size[0] * 1), int(img.size[1] * 1))
        img = img.resize(size=new_dim)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(pixels_path).convert("L")
        mask = mask.resize(size=new_dim, resample=Image.NEAREST)

        mask = np.array(mask)
        mask = 255 - np.array(mask)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        if self.transforms:
            img, masks = get_transform(img, masks, idx)

        if self.to_tensor:
            img = F.pil_to_tensor(img)
            img = F.convert_image_dtype(img)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd


        return img, target

    def __len__(self):
        return len(self.imgs)

# %% TEST IMAGES
imgs = list(sorted(os.listdir(os.path.join(path_to_dataset, SCANS_DIR))))
masks_pixels = list(sorted(os.listdir(os.path.join(path_to_dataset, "ground-truth-pixel/ground-truth-pixel"))))

idx = 3
# load images ad masks
img_path = os.path.join(path_to_dataset, SCANS_DIR, imgs[idx])
print(img_path)
pixels_path = os.path.join(path_to_dataset, "ground-truth-pixel/ground-truth-pixel", masks_pixels[idx])
img = Image.open(img_path).convert("RGB")

new_dim = (int(img.size[0] * 1), int(img.size[1] * 1))
img = img.resize(size=new_dim)
mask = Image.open(pixels_path).convert("L")
mask = mask.resize(size=new_dim, resample=Image.NEAREST)
mask = np.array(mask)
mask = 255 - np.array(mask)
img, mask = get_transform(img, Image.fromarray(mask), idx)

obj_ids = np.unique(mask)
obj_ids = obj_ids[1:]

masks = np.array(mask) == obj_ids[:, None, None]

# get bounding box coordinates for each mask
num_objs = len(obj_ids)
boxes = []
for i in range(num_objs):
    pos = np.where(masks[i])
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    boxes.append([xmin, ymin, xmax, ymax])

boxes = torch.as_tensor(boxes, dtype=torch.float32)
# there is only one class
labels = torch.ones((num_objs,), dtype=torch.int64)
masks = torch.as_tensor(masks, dtype=torch.uint8)

image_id = torch.tensor([idx])
area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
# suppose all instances are not crowd
iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

if self.transforms:
    img, masks = get_transform(img, masks, idx)

if self.to_tensor:
    img = F.pil_to_tensor(img)
    img = F.convert_image_dtype(img)



# %% DELETE EXCESS FILES # %%
exces_scans = list(sorted(os.listdir(os.path.join(path_to_dataset, "scans/scans"))))
del_files = [os.path.join(path_to_dataset, "scans/scans/") + s for s in exces_scans if re.search(r'(.*?4[0-9][0-9].*?)', s)]
del_files = del_files[1:]

[os.remove(file) for file in del_files]

# %% CLEAR PICTURES WITHOUT MASK

imgs = list(sorted(os.listdir(os.path.join(path_to_dataset, "scans/scans"))))
pixel_mask = list(sorted(os.listdir(os.path.join(path_to_dataset, "ground-truth-pixel/ground-truth-pixel"))))
masks = list(sorted(os.listdir(os.path.join(path_to_dataset, "ground-truth-maps/ground-truth-maps"))))

for idx in range(len(imgs)):
    print(idx)
    img_path = os.path.join(path_to_dataset, "scans/scans", imgs[idx])
    mask_path = os.path.join(path_to_dataset, "ground-truth-maps/ground-truth-maps", masks[idx])
    pixel_mask_path = os.path.join(path_to_dataset, "ground-truth-pixel/ground-truth-pixel", pixel_mask[idx])
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path)

    mask = np.array(mask)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[1:]
    num_objs = len(obj_ids)

    list_to_delete = ["037", "217", "219", "224", "387", "392", "400",
                      "398", "396", "394", "393", "384", "382", "381", "379",
                      "376", "374", "373", "370", "367", "355", "333", "324",
                      "320", "237", "236", "235", "234", "233", "232", "229",
                      "228", "227", "226", "225", "224", "223", "222", "221",
                      "220", "216", "215", "214", "213", "138", "053", "038"]

    if (num_objs == 0 or
        any([re.search(r'(.*?00' + file_name + '.*?)', img_path)
            for file_name in list_to_delete])):
        os.remove(img_path)
        os.remove(mask_path)
        os.remove(pixel_mask_path)



 # %%

dataset = StaverDataset(path_to_dataset)
dataset[293]


# %%
# dataset = StaverDataset(path_to_dataset, get_transform(train=False))
# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn
# )
#
# a = iter(data_loader)
# for idx in range(len(dataset)):
#   image = next(a)
#   print(image)


# %%

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = get_instance_segmentation_model(2)
dataset = StaverDataset(path_to_dataset, transforms = True, to_tensor= True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn
)

# %%

images, targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images, targets)  # Returns losses and detections

# %%
im = T.ToPILImage()(targets[0]["masks"]).convert("RGB")
display(im)

# %% For inference

# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)  # Returns predictions

# %%

# use our dataset and defined transformations
dataset = StaverDataset(path_to_dataset, transforms = True, to_tensor= True)
dataset_test = StaverDataset(path_to_dataset, transforms = False, to_tensor= True)

# %%

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
# dataset = torch.utils.data.Subset(dataset, indices)
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# %%
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2, shuffle=True, num_workers=0, collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn,
)

# %%

"""Now let's instantiate the model and the optimizer"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# %%

num_epochs = 1
metric_logger_iter = {"epoch": [], "logs": []}
loss_dict_iter = []

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    metric_logger, loss_dict_output = train_one_epoch(model, optimizer,
                                                      data_loader,
                                                      device, epoch,
                                                      print_freq=1)

    loss_dict_iter.append(loss_dict_output)
    metric_logger_iter["epoch"].append(epoch)
    metric_logger_iter["logs"].append(metric_logger)

    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

# %% DELETE
epoch = 0
model.train()
metric_logger = utils.MetricLogger(delimiter="  ")
metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
header = f"Epoch: [{epoch}]"

lr_scheduler = None
if epoch == 0:
    warmup_factor = 1.0 / 1000
    warmup_iters = min(1000, len(data_loader) - 1)

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    )


loss_dict_output = []
iterace = 0
print_freq = 1
scaler = None

images, targets = next(metric_logger.log_every(data_loader, print_freq, header))
for images, targets in metric_logger.log_every(data_loader, print_freq, header):
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    with torch.cuda.amp.autocast(enabled=scaler is not None):
        loss_dict = model(images, targets)
        del loss_dict["loss_classifier"]
        del loss_dict["loss_rpn_box_reg"]
        del loss_dict["loss_objectness"]
        losses = sum(loss for loss in loss_dict.values())

    # reduce losses over all GPUs for logging purposes
    loss_dict_reduced = utils.reduce_dict(loss_dict)
    losses_reduced = sum(loss for loss in loss_dict_reduced.values())

    loss_value = losses_reduced.item()

    loss_dict_output.append({"iterace": iterace,
                             "logs": {loss[0]:float(loss[1]) for loss
                                      in loss_dict_reduced.items()}})

    if not math.isfinite(loss_value):
        print(f"Loss is {loss_value}, stopping training")
        print(loss_dict_reduced)
        sys.exit(1)

    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        losses.backward()
        optimizer.step()

    if lr_scheduler is not None:
        lr_scheduler.step()

    metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    iterace += 1


# %%
# """Now that training has finished, let's have a look at what it actually predicts
# in a test image"""
#
# # pick one image from the test set
# img, _ = dataset_test[0]
# # put the model in evaluation mode
# model.eval()
# with torch.no_grad():
#     prediction = model([img.to(device)])
#
# prediction
#
# Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()).show()
#
# Image.fromarray(prediction[0]["masks"][0, 0].mul(255).byte().cpu().numpy()).show()

# %% TEST KB DATA

model_checkpoint_path = "./tmp/saved_metrics/saved_pixels_feature_extra/checkpoint.pth"

num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

state_dict = torch.load(model_checkpoint_path, map_location=torch.device('cpu'))
print(state_dict.keys())

model.load_state_dict(state_dict)


# %%

len(os.listdir("../data/kb_invoices/"))

for kb_pic in range(4):

    img = Image.open("../data/kb_invoices/" + os.listdir("../data/kb_invoices/")[kb_pic])
    img = torchvision.transforms.functional.pil_to_tensor(img)
    img = img/255

    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    prediction

    Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()).show()
    Image.fromarray(prediction[0]["masks"][0, 0].mul(255).byte().cpu().numpy()).show()

# %%

# %%
path_to_kb_data = "../data/kb_invoices/"

scan_kb_files = glob.glob(path_to_kb_data + '*.png')
scan_kb_files = sorted(scan_kb_files)

# %%
steps = len(scan_kb_files)
checkpoints = model_checkpoint_path
rows = len(checkpoints) + 1

# %%
plt.figure(figsize=(steps * 10, rows * 10))
for i in range(steps):
    plt.subplot(rows, steps, i + 1)
    plt.imshow(imread(scan_kb_files[i]))


for kb_pic in range(4):

    img = Image.open("../data/kb_invoices/" + os.listdir("../data/kb_invoices/")[kb_pic])
    img = torchvision.transforms.functional.pil_to_tensor(img)
    img = img/255

    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])


for i, c in enumerate(checkpoints):
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    predicted = prediction[0]["masks"][0, 0].mul(255).byte().cpu().numpy()
    for s in range(steps):
        plt.subplot(rows, steps, i * steps + s + steps + 1)
        plt.imshow(predicted[s])

plt.show()