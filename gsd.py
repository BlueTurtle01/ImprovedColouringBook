# pip install keras-segmentation
from keras_segmentation.pretrained import pspnet_50_ADE_20K

model = pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset

# load any of the 3 pretrained models

out = model.predict_segmentation(
    inp="MultiInputs/horse.jpg",
    out_fname="out.png"
)

