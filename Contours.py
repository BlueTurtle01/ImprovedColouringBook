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


def kmeans(clusters):
    file_name = "mia"
    file_path = str("InputImages/" + file_name + ".jpg")

    #Read the image
    imag = Image.open(file_path)
    imshow(imag)

    #Dimension of the original image
    cols, rows = imag.size

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


def canny(img):
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(original_img, (15, 15), sigmaX=1)

    edged = auto_canny(blurred)
    edged = cv2.dilate(edged, (5, 5), iterations=1)
    edged = cv2.erode(edged, (5, 5), iterations=1)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    edged = cv2.bitwise_not(edged)
    rgb = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
    print("Number of Contours found = " + str(len(contours)))

    final_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            final_contours.append(contour)
            pass
        else:
            pass

    temp_array = np.ones([rgb.shape[0], rgb.shape[1], rgb.shape[2]])
    contours_ = cv2.drawContours(temp_array, final_contours, -1, (0, 0, 0), thickness=1)
    plt.imshow(contours_)
    plt.show()






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


img, palette = kmeans(int(5))
canny(img)
