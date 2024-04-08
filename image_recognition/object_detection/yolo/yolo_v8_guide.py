from ultralytics import YOLO
from PIL import Image
import cv2

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
# Train mode is used for training a YOLOv8 model on a custom dataset. In this mode, the model is
# trained using the specified dataset and hyperparameters. The training process involves
# optimizing the model's parameters so that it can accurately predict the classes and locations
# of objects in an image.
results = model.train(data='coco128.yaml', epochs=3)
results = model.train(data='coco128.yaml', epochs=3, device='mps')
results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device='mps')

# Evaluate the model's performance on the validation set
# Val mode is used for validating a YOLOv8 model after it has been trained. In this mode,
# the model is evaluated on a validation set to measure its accuracy and generalization
# performance. This mode can be used to tune the hyperparameters of the model to improve its
# performance.
results = model.val()

# Perform object detection on an image using the model
results = model('https://ultralytics.com/images/bus.jpg')

# capture camera
# import cv2
# cap=cv2.VideoCapture(0)
# while True:
#     ret,img=cap.read()
#     cv2.imshow('webcam',img)
#     k=cv2.waitKey(10)
#     if k==27:
#         break;
# cap.release()
# cv2.destroyAllWindows()

# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
results = model.predict(source="0", show=True)
results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open("bus.jpg")
results = model.predict(source=im1, save=True)  # save plotted images

# from ndarray
im2 = cv2.imread("bus.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# from list of PIL/ndarray
results = model.predict(source=[im1, im2])



# Export the model to ONNX format
success = model.export(format='onnx')