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

def gaussian_filter(img):
    k = 7
    blurred = cv2.GaussianBlur(img, (k, k), sigmaX=(k-1)/6)
    return blurred



def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def create_trackbar(blurred, dst):
    def callback(x):
        print(x)

    cv2.namedWindow('image')  # make a window with name 'image'
    cv2.createTrackbar('L', 'image', 0, 255, callback)  # lower threshold trackbar for window 'image
    cv2.createTrackbar('U', 'image', 0, 255, callback)  # upper threshold trackbar for window 'image

    while (1):
        numpy_horizontal_concat = np.concatenate((blurred, dst), axis=1)  # to display image side by side
        cv2.imshow('image', numpy_horizontal_concat)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # escape key
            break
        l = cv2.getTrackbarPos('L', 'image')
        u = cv2.getTrackbarPos('U', 'image')

        dst = cv2.Canny(blurred, l, u)

    cv2.destroyAllWindows()


def canny(img, clusters):
    """
    :param img: input image as an array
    :param clusters: number of k-means clusters
    :return: output image after clustering and edge detection has been applied
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = gaussian_filter(img)

    """
    Any pixels with a value below the minval are defined as definitely not edges.
    Any pixels with a value above the maxval are defined as definitely edges and hence are retained.
    If we set maxval too high then we discard a lot of lighter pixels as not edges.
    If we set minval too high then we also discard a lot of darker pixels as not edges.
    """

    #dst = cv2.Canny(blurred, lower, upper)
    #create_trackbar(blurred, dst)

    dst = auto_canny(blurred)
    rgb = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)

    #rgb = cv2.bitwise_not(rgb) # Credit: https://stackoverflow.com/a/40954142/4367851
    # Save and display output image
    #rgb = cv2.morphologyEx(rgb, cv2.MORPH_OPEN, (5, 5))
    rgb = cv2.dilate(rgb, (20, 20), iterations=1)
    rgb = cv2.erode(rgb, (5, 5), iterations=1)
    rgb = cv2.bitwise_not(rgb)  # Credit: https://stackoverflow.com/a/40954142/4367851
    imsave(("OutputImages/" + str(file_name) + str(clusters) + "Canny.jpg"), rgb)

    return rgb


def plot_creator(clusters):
    """
    :param n: number of clusters for k-means
    :return: NA - plots
    """
    img, palette = kmeans(int(clusters))

    fig, axs = plt.subplots(3, 2, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.3, wspace=.001)
    axs = axs.ravel()

    for i in range(0, 6):
        if i % 6 == 0:
            img = imread(("OutputImages/" + str(file_name) + str(clusters) + "Clustered.jpg"))
            axs[0].imshow(img.astype(np.uint8))
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            axs[0].set_title("Original with " + str(clusters) + " clusters")

        else:
            axs[i].imshow(canny(img, clusters))
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_title(str("Canny Kernel: " + str(i)))

    plt.savefig("Output" + str(clusters) + ".jpg", dpi=600, bbox_inches='tight')
    plt.show()


plot_creator(clusters=10)
