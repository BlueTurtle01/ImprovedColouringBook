from matplotlib.pyplot import imread, imshow, imsave
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import pandas as pd
import os


def find_colour_pal(compressed_image):
    # Credit: https://stackoverflow.com/a/51729498/4367851
    temp_image = Image.fromarray(np.uint8(compressed_image)).convert('RGB')
    colours = Image.Image.getcolors(temp_image)
    df = pd.DataFrame()
    for i in range(len(colours)):
        count, pixel = colours[i]
        red, green, blue = pixel
        hex = '#%02x%02x%02x' % (int(red), int(green), int(blue))
        df = df.append({"Count": int(count), "RGB": pixel, "HEX": hex}, ignore_index=True)

    RGB = df["RGB"].tolist()
    palette = np.array(RGB)[np.newaxis, :, :]
    # #palette = palette.reshape(2, -1, 3)
    # plt.figure(figsize=(20, 10))
    # plt.imshow(palette)
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig("Name.png", dpi=600, bbox_inches='tight')
    # plt.show()

    return palette

def plot_histogram(imag):
    plt.hist(imag.ravel(), 256) #Histogram is not bimodal so we can not use Otsu's thresholding.
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Histogram for input image")
    plt.show()


def plot_image(image, title, map="gray"):
    plt.imshow(image, cmap=map)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


def kmeans(clusters):
    file_name = "paul"
    file_path = str("InputImages/" + file_name + ".jpg")

    #Read the image
    #imag = Image.open(file_path)
    imag = cv2.imread(file_path)
    plot_image(image=cv2.cvtColor(imag, cv2.COLOR_BGR2RGB), title="Original Image")
    #plot_histogram(imag)

    #Dimension of the original image
    #cols, rows = imag.size
    rows, cols, _ = imag.shape


    #Flatten the image
    imag = np.array(imag).reshape(rows*cols, 3)

    #Implement k-means clustering to form k clusters
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(imag)

    #Replace each pixel value with its nearby centroid
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)


    #Reshape the image to original dimension
    compressed_image = compressed_image.reshape(rows, cols, 3)
    palette = find_colour_pal(compressed_image)


    #Save and display output image
    imsave(("OutputImages/" + str(file_name) + str(clusters) + "Clustered.jpg"), compressed_image)
    #imshow(compressed_image)
    #plt.show()

    return compressed_image, palette


def auto_canny(image, sigma=0.5):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def blend_outputs(outline):
    alpha = 0.4 #This is the amount of the coloured image we include. 0 is none.
    beta = 1 - alpha
    compressed_image = imread("OutputImages/paul10Clustered.jpg")
    compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)

    outline = np.concatenate([outline[..., np.newaxis]] * 3, axis=2)
    output = cv2.addWeighted(outline, beta, compressed_image, alpha, 0.0)

    plt.imshow(output)
    plt.title("Blended")
    plt.xticks([])
    plt.yticks([])
    plt.show()

def canny(img):
    compressed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #imread reads images as Blue, Green, Red, Alpha
    plot_image(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), title="Compressed")

    k = 15
    blurred = cv2.GaussianBlur(compressed_img, (k, k), sigmaX=(k-1)/6)
    plot_image(image=blurred, title="Blurred", map="gray")

    th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 3)
    th3 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, (9, 9))

    blend_outputs(th3)
    plot_image(th3, title="Threshold Gray")

    th3 = cv2.bitwise_not(th3)

    edged = auto_canny(th3)
    edged = cv2.dilate(edged, (20, 20), iterations=1)
    edged = cv2.erode(edged, (20, 20), iterations=1)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    edged = cv2.bitwise_not(edged)
    rgb = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
    print("Number of Contours found = " + str(len(contours)))

    final_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 40:
            final_contours.append(contour)
            pass
        else:
            pass

    temp_array = np.ones([rgb.shape[0], rgb.shape[1], rgb.shape[2]])
    contours_ = cv2.drawContours(temp_array, final_contours, -1, (0, 0, 0), thickness=3)
    plot_image(contours_, title="Contours")

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # lines0 = cv2.drawContours(rgb, contours, 0, (0, 0, 0), 3)
    # #imsave("OutputImages/" + str(file_name) + ".jpg", lines0)
    # ax1.imshow(lines0)
    # ax1.set_title("Level 0 Contours")
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    #
    # lines1 = cv2.drawContours(rgb, contours, 1, (0, 0, 0), 3)
    # ax2.imshow(lines1)
    # ax2.set_title("Level 1 Contours")
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    #
    # lines2 = cv2.drawContours(rgb, contours, 2, (0, 0, 0), 3)
    # ax3.imshow(lines2)
    # ax3.set_title("Level 2 Contours")
    # ax3.set_xticks([])
    # ax3.set_yticks([])
    #
    # linesAll = cv2.drawContours(rgb, contours, -1, (0, 0, 0), 3)
    # ax4.imshow(linesAll)
    # ax4.set_title("All Contours")
    # ax4.set_xticks([])
    # ax4.set_yticks([])
    # plt.savefig("ContoursOutput.jpg", dpi = 400, bbox_inches='tight')
    # plt.show()

    return rgb


img, palette = kmeans(int(10))
canny(img)
