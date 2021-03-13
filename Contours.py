from matplotlib.pyplot import imread, imshow, imsave
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Clustering import file_name, kmeans
from PIL import Image
import os



def filter2d(img, n):
    kernel = np.ones((n, n), np.float32)/n**2
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred


def canny(img, n):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #blurred = filter2d(img, n)
    lower = 100
    upper = 200

    edged = cv2.Canny(img, lower, upper)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    plt.imshow(edged)

    rgb = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)

    print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    # -1 signifies drawing all contours
    rgb = cv2.drawContours(rgb, contours, -1, (0, 255, 0), 3)

    plt.imshow(rgb)
    plt.show()

    return rgb


img = kmeans(int(30))
canny(img, 3)
