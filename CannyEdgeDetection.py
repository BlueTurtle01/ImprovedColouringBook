from matplotlib.pyplot import imread, imshow, imsave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from Clustering import file_name, kmeans



def filter2d(img, n):
    kernel = np.ones((n, n), np.float32)/n**2
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred


def canny(img, n):
    blurred = filter2d(img, n)
    lower = 10
    upper = 100

    """
    Any pixels with a value below the minval are defined as definitely not edges.
    Any pixels with a value above the maxval are defined as definitely edges and hence are retained.
    If we set maxval too high then we discard a lot of lighter pixels as not edges.
    If we set minval too high then we also discard a lot of darker pixels as not edges.
    """

    dst = cv2.Canny(blurred, lower, upper)
    rgb = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)

    rgb = cv2.bitwise_not(rgb) # Credit: https://stackoverflow.com/a/40954142/4367851
    # Save and display output image
    imsave(("OutputImages/" + str(file_name) + "CannyOutput.jpg"), rgb)

    return rgb


def plot_creator():
    max = 3
    for j in range(1, max):
        img = kmeans(int(j*10))
        height, width, _ = img.shape
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fig, axs = plt.subplots(max, 6, figsize=(15, 6), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.5, wspace=.001)
        axs = axs.ravel()
        for i in range(0, 6*max):
            if i % 6 == 0:
                img = imread(("OutputImages/" + str(file_name) + str(j*10) + "Clustered.jpg"))
                axs[i].imshow(img.astype(np.uint8))
                axs[i].set_xticks([])
                axs[i].set_yticks([])
            else:
                axs[i].imshow(canny(img, i*2))
                axs[i].set_xticks([])
                axs[i].set_yticks([])

    plt.show()




def plot_creator2():
    for j in range(1, 4):
        fig, axs = plt.subplots(1, 6, figsize=(15, 6), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=.5, wspace=.001)
        axs = axs.ravel()

        for i in range(0, 6):
            if i % 6 == 0:
                img = imread(("OutputImages/" + str(file_name) + str(j*10) + "Clustered.jpg"))
                axs[0].imshow(img.astype(np.uint8))
                axs[0].set_xticks([])
                axs[0].set_yticks([])
            else:
                img = kmeans(int(i * 10))
                axs[i].imshow(canny(img, i*2))
                axs[i].set_xticks([])
                axs[i].set_yticks([])
        plt.show()

plot_creator2()