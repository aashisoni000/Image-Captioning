import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Pre-trained Faster R-CNN Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  

image_path = r'C:\Users\Admin\Documents\code\image_captioning_project\cat.jpg'  
image = Image.open(image_path).convert("RGB")  
image_tensor = F.to_tensor(image)  


with torch.no_grad():  
    predictions = model([image_tensor])  


boxes = predictions[0]['boxes'].numpy() 
labels = predictions[0]['labels'].numpy()  
scores = predictions[0]['scores'].numpy()  

threshold = 0.5
boxes = boxes[scores >= threshold]
labels = labels[scores >= threshold]
scores = scores[scores >= threshold]

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


fig, ax = plt.subplots(1)
ax.imshow(image)


for box, label, score in zip(boxes, labels, scores):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

  
    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

  
    label_text = f"{COCO_CLASSES[label]}: {score:.2f}"
    ax.text(x1, y1 - 5, label_text, color='r', fontsize=12, backgroundcolor='white')

plt.axis('off')  
plt.title("Object Detection with Faster R-CNN")
plt.show()