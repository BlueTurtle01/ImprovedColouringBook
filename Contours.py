from matplotlib.pyplot import imread, imshow, imsave
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import pandas as pd
import os

file_name = "astrocute"
file_path = str("InputImages/" + file_name + ".jpg")


def find_colour_pal(compressed_image, im_name):
    """
    This function draws a grid of the colours used for the clusters. This gives the user an easy way to check they have the correct colouring crayons.

    :param compressed_image: Numpy array of the pixel values for the compressed image after applying k-means to the original imput image.
    :param im_name: Name of the file to be used as the filename when saving.
    :return: A palette of the cluster colours
    """
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

    if palette.shape[0] % 2 == 0:
        palette = palette.reshape(2, -1, 3)
    elif palette.shape[0] % 3 == 0:
        palette = palette.reshape(3, -1, 3)

    plt.figure(figsize=(20, 10))
    plt.imshow(palette)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(("OutputImages/" + str(im_name) + "/" + "Palette.png"), dpi=600, bbox_inches='tight')
    plt.show()

    return palette


def plot_histogram(imag):
    plt.hist(imag.ravel()/255., 256)  # Histogram is not bimodal so we can not use Otsu's thresholding.
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Histogram for input image")
    plt.show()


def plot_image(image, title, im_name, map="gray"):
    plt.imshow(image, cmap=map)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    imsave(("OutputImages/" + str(im_name) + "/" + str(title) + ".jpg"), image, cmap=map)
    plt.show()


def kmeans(clusters):
    # Read the image
    imag = cv2.imread(file_path)
    plot_image(image=cv2.cvtColor(imag, cv2.COLOR_BGR2RGB), title="Original Image", im_name=file_name)
    plot_histogram(imag)

    # Dimension of the original image
    rows, cols, _ = imag.shape

    # Flatten the image
    imag = np.array(imag).reshape(rows*cols, 3)

    # Implement k-means clustering to form k clusters
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(imag)

    #  Replace each pixel value with its nearby centroid
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

    #  Reshape the image to original dimension
    compressed_image = compressed_image.reshape(rows, cols, 3)
    palette = find_colour_pal(compressed_image, im_name=file_name)

    # Save and display output image
    if os.path.isdir("OutputImages/" + str(file_name)):
        pass
    else:
        os.mkdir("OutputImages/" + str(file_name))
    imsave(("OutputImages/" + str(file_name) + "/" + str(file_name) + str(clusters) + "Clustered.jpg"), compressed_image)

    return compressed_image, palette


def auto_canny(image, sigma=0.5):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def blend_outputs(outline, clusters):
    """
    Some of the edges are not closed loops so it can be confusing where one colour stops and another starts.
    I think overlaying the outlines onto a slightly transparent version of the compressed image gives a good guide to the user for which colour to fill each area with.

    :param outline: The numpy array of outline pixel values
    :param clusters: Number of clusters from the k-means algorithm
    :return: NA
    """
    alpha = 0.4  # This is the amount of the coloured image we include. 0 is none.
    beta = 1 - alpha
    compressed_image = imread(("OutputImages/" + str(file_name) + "/" + str(file_name) + str(clusters) + "Clustered.jpg"))
    compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)

    outline = np.concatenate([outline[..., np.newaxis]] * 3, axis=2)
    output = cv2.addWeighted(outline, beta, compressed_image, alpha, 0.0)

    plt.imshow(output)
    plt.title("Blended")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def draw_contours(edged):
    """
    Draw the contours over a blank array. The function cv2.DrawContours overlays the contours on top of the bitwise array.
    Which is not ideal if the bitwise array contains some small, noisy contours. Therefore, I created an empty array first and then used this as the base
    for drawing the contours onto.
    :param edged: Output of the Canny Edge Detection algorithm after applying erode and dilation.
    :return:
    """
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

    plot_image(contours_, title="Contours", im_name=file_name)


def main_func(img, clusters):
    """

    :param img:
    :param clusters: Number of clusters
    :return: NA
    """

    # The output of the k-means algorithm is an array in the form of Blue,Green,Red (BGR) which needs to be converted to grayscale before we conduct a Gaussian Blur.
    compressed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # imread reads images as Blue, Green, Red, Alpha
    plot_image(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), title="Compressed", im_name=file_name)

    k = 15  # Size of the kernal for the Gaussian Blur
    blurred = cv2.GaussianBlur(compressed_img, (k, k), sigmaX=(k-1)/6)
    plot_image(image=blurred, title="Blurred", im_name=file_name, map="gray")

    # Conduct Adaptive Thresholding to the blurred image
    threshold_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 3)
    threshold_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, (9, 9))

    # Overlay the threshold_image on top of a slightly transparent copy of the clustered image.
    blend_outputs(threshold_image, clusters=clusters)
    plot_image(threshold_image, title="Threshold Gray", im_name=file_name)

    inverted_threshold = cv2.bitwise_not(threshold_image)

    edged = auto_canny(threshold_image)
    edged = cv2.dilate(edged, (20, 20), iterations=1)
    edged = cv2.erode(edged, (20, 20), iterations=1)

    draw_contours(edged)

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


def calculate_clusters():
    # Credit: https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
    imag = cv2.imread(file_path)
    plot_image(image=cv2.cvtColor(imag, cv2.COLOR_BGR2RGB), title="Original Image", im_name=file_name)
    plot_histogram(imag)

    Sum_of_squared_distances = []
    # Dimension of the original image
    rows, cols, _ = imag.shape

    # Flatten the image
    imag = np.array(imag).reshape(rows*cols, 3)

    K = range(1, 5)
    for k in K:
        km = KMeans(n_clusters=k)
        km.fit(imag)
        Sum_of_squared_distances.append(km.inertia_)

    error_df = pd.DataFrame(list(zip(K, Sum_of_squared_distances)), columns=["K", "Error"])
    error_df["Difference"] = error_df["Error"] - error_df["Error"].shift(-1)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.plot(K, error_df["Difference"])
    #plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    try:
        clusters = min(error_df.index[error_df["Difference"] < 3e+06].tolist())
    except ValueError:
        clusters = max(error_df["K"])

    # Implement k-means clustering to form k clusters
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(imag)

    #  Replace each pixel value with its nearby centroid
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

    #  Reshape the image to original dimension
    compressed_image = compressed_image.reshape(rows, cols, 3)
    find_colour_pal(compressed_image, im_name=file_name)

    # Save and display output image
    if os.path.isdir("OutputImages/" + str(file_name)):
        pass
    else:
        os.mkdir("OutputImages/" + str(file_name))
    imsave(("OutputImages/" + str(file_name) + "/" + str(file_name) + str(clusters) + "Clustered.jpg"), compressed_image)

    try:
        return min(error_df.index[error_df["Difference"] < 3e+06].tolist())
    except ValueError:
        return max(error_df["K"])

clusters = calculate_clusters()

img, palette = kmeans(int(clusters))
main_func(img, clusters=clusters)

