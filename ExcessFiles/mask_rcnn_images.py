import torch
import torchvision
import cv2
from PIL import Image
from utils import draw_segmentation_map, get_outputs
import matplotlib.pyplot as plt
from torchvision.transforms import transforms as transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# initialize the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)

# load the model on to the computation device and set to eval mode
model.eval()


# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

image = Image.open("MultiInputs/david.jpg").convert('RGB')
# keep a copy of the original image for OpenCV functions and applying masks
orig_image = image.copy()
# transform the image
image = transform(image)
# add a batch dimension
image = image.unsqueeze(0)
masks, boxes, labels = get_outputs(image, model, 0.965)
result = draw_segmentation_map(orig_image, masks, boxes, labels)
# visualize the image
plt.imshow('Segmented image', result)
plt.show()
# set the save path
save_path = "MultiOutput/davidMask.jpg"
cv2.imwrite(save_path, result)

