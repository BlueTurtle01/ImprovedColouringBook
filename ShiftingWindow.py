from matplotlib.pyplot import imread, imshow, imsave
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from itertools import chain
from Clustering import file_name
from CannyEdgeDetection import filter2d


original_image = mpimg.imread(("InputImages/" + str(file_name) + ".jpg"))
img = mpimg.imread(("OutputImages/" + str(file_name) + "Clustered.jpg"))
height, width, _ = img.shape

window_size = 2
rows = int(height / window_size)
columns = int(width / window_size)

filter2d(img, 5)

# Create a new image that we will reassign colours to.
new_image = np.empty([rows, columns, 3])
for row in range(new_image.shape[0] - 1):
    rowstart = row * window_size
    rowend = rowstart + window_size
    for col in range(new_image.shape[1] - 1):
        colstart = col * window_size
        colend = colstart + window_size
        img_ = img[rowstart:rowend, colstart:colend]
        rowitems = img_[0].tolist()
        colitems = img_[1].tolist()
        newlist = list(chain.from_iterable([rowitems, colitems]))
        new_list = [tuple(l) for l in newlist]

        if new_list[0] == new_list[1] == new_list[2] == new_list[3]:
            new_image[row, col, ] = 255
            new_image[row+1, col+1, ] = 255
            new_image[row, col+1, ] = 255
            new_image[row+1, col, ] = 255
            pass
        else: #If the pixels are different, it must be a boundary, therefore replace all with black
            new_image[row, col, ] = 0
            new_image[row+1, col+1, ] = 0
            new_image[row, col+1, ] = 0
            new_image[row+1, col, ] = 0


fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow((original_image).astype(np.uint8))
ax1.set_title("Original")
ax1.set_xticks([])
ax1.set_yticks([])
ax2.imshow(img.astype(np.uint8))
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("Clustered")
ax3.imshow(new_image.astype(np.uint8))
ax3.set_title("Final")
ax3.set_xticks([])
ax3.set_yticks([])
plt.show()

imsave(("OutputImages/" + str(file_name) + "Window.jpg"), new_image.astype(np.uint8))