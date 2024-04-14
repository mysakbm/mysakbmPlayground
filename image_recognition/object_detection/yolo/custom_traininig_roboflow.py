import os
import cv2
from ultralytics import YOLO

os.chdir("image_recognition/object_detection/yolo")

# In[] Check the bounding boxes
image_idx = 2

image_path = '../datasets/signatures_dataset/SignverOD/' + 'images/'
label_path = '../datasets/signatures_dataset/SignverOD/' + 'labels/'

image_list = os.listdir(image_path)

with open(label_path + image_list[image_idx].split(".")[0] + ".txt", "r") as file:
    L = file.readlines()

x, y, w, h = [float(x) for x in L[0].strip().split()[1:]]
x = x - w / 2
y = y - h / 2

image = cv2.imread(image_path + image_list[image_idx])
image_w = image.shape[1]
image_h = image.shape[0]

x = int(x * image_w)
y = int(y * image_h)
w = int(w * image_w)
h = int(h * image_h)

cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# In[] Train the model

# Path to the dataset
dataset_path = '../datasets/signatures_dataset/SignverOD/'

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model.info() # display architecture and information

# Train the model with 2 GPUs
results = model.train(
    data=dataset_path + '/signverOD.yaml',
    # freeze=21,
    imgsz=96,
    epochs=1,
    batch = 4,
    device='mps',
    degrees=35,
    shear=15,
    flipud=0.5,
    copy_paste=0.1
)

# results = model.val(
#     data='../datasets/signatures_dataset/SignverOD/signverOD.yaml',
#     # batch = 8,
#     # device='mps'
# )
