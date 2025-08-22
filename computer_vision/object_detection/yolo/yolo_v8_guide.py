from ultralytics import YOLO
from ultralytics import settings
from ultralytics.utils.benchmarks import benchmark
from ultralytics import Explorer

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

from utils import time_run

os.chdir("computer_vision/object_detection/yolo")

# # View all settings
# print(settings)
# # Return a specific setting
# settings['runs_dir']
# # Update a setting
# settings.update({'runs_dir': '/path/to/runs'})
# # Update multiple settings
# settings.update({'runs_dir': '/path/to/runs', 'tensorboard': False})
# # Reset settings to default values
# settings.reset()

# Create a new YOLO model from scratch
# model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')  # load an official detection model
# model = YOLO('yolov8n-seg.pt')  # load an official segmentation model
# model = YOLO('path/to/best.pt')  # load a custom model

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='coco128.yaml', epochs=1)
results = model.train(data='coco128.yaml', epochs=1, device='mps')
results = model.train(data='coco128.yaml', epochs=1, imgsz=640, device='mps')

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model('https://ultralytics.com/images/bus.jpg')

# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
results = model.predict(source="0", show=True)
results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open("bus.jpg")
results = model.predict(source=im1, save=True)  # save plotted images
cv2.imshow('image', np.array(im1))
cv2.waitKey(0)
cv2.destroyAllWindows()


# from ndarray
im2 = cv2.imread("bus.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# from list of PIL/ndarray
results = model.predict(source=[im1, im2])


# # Track with the model
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")

# Benchmark
benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device='cpu')

# create an Explorer object
exp = Explorer(data='coco128.yaml', model='yolov8n.pt')
exp.create_embeddings_table()

similar = exp.get_similar(img='https://ultralytics.com/images/bus.jpg', limit=10)
print(similar.head())

# Search using multiple indices
similar = exp.get_similar(
                        img=['https://ultralytics.com/images/bus.jpg',
                             'https://ultralytics.com/images/bus.jpg'],
                        limit=10
                        )
print(similar.head())

''' python
from ultralytics.models.yolo import DetectionTrainer, DetectionValidator, DetectionPredictor

# trainer
trainer = DetectionTrainer(overrides={})
trainer.train()
trained_model = trainer.best

# Validator
val = DetectionValidator(args=...)
val(model=trained_model)

# predictor
pred = DetectionPredictor(overrides={})
pred(source=SOURCE, model=trained_model)

# resume from last weight
overrides["resume"] = trainer.last
trainer = detect.DetectionTrainer(overrides=overrides)
'''








