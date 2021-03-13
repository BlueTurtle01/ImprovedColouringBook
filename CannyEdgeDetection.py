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


def canny(img, n, k):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = filter2d(img, n)
    lower = 100
    upper = 200

    """
    Any pixels with a value below the minval are defined as definitely not edges.
    Any pixels with a value above the maxval are defined as definitely edges and hence are retained.
    If we set maxval too high then we discard a lot of lighter pixels as not edges.
    If we set minval too high then we also discard a lot of darker pixels as not edges.
    """

    dst = cv2.Canny(blurred, lower, upper)
    rgb = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)

    #rgb = cv2.bitwise_not(rgb) # Credit: https://stackoverflow.com/a/40954142/4367851
    # Save and display output image
    #rgb = cv2.morphologyEx(rgb, cv2.MORPH_OPEN, (5, 5))
    rgb = cv2.dilate(rgb, (5, 5), iterations=1)
    rgb = cv2.bitwise_not(rgb)  # Credit: https://stackoverflow.com/a/40954142/4367851
    imsave(("OutputImages/" + str(file_name) + str(n*10) + "Kmeans" + str(k) + "Canny.jpg"), rgb)

    return rgb


def plot_creator(n):
    fig, axs = plt.subplots(3, 2, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.3, wspace=.001)
    axs = axs.ravel()

    img = kmeans(int(n * 10))
    for i in range(0, 6):
        if i % 6 == 0:
            img = imread(("OutputImages/" + str(file_name) + str(n*10) + "Clustered.jpg"))
            axs[0].imshow(img.astype(np.uint8))
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            if n == 1:
                axs[0].set_title("Original with " + str(n) + " cluster.")
            else:
                axs[0].set_title("Original with " + str(n) + " clusters.")

        else:
            axs[i].imshow(canny(img, i*2, n))
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_title(str("Canny Kernel: " + str(i)))

    plt.savefig("Output" + str(n) + ".jpg")
    plt.show()

for n in range(1,5):
    plot_creator(n)
