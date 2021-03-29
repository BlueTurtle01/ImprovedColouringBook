#Credit: https://towardsdatascience.com/image-compression-using-k-means-clustering-aa0c91bb0eeb
# Credit: https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_imgproc/py_filtering/py_filtering.html
from matplotlib.pyplot import imread, imshow, imsave
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd



file_name = "ash"
file_path = str("InputImages/" + file_name + ".jpg")


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


def kmeans(clusters, file_path=file_path):
    #Read the image
    #image = imread(file_path)
    imag = Image.open(file_path)
    imshow(imag)

    #Dimension of the original image
    #rows = image.shape[0]
    #cols = image.shape[1]
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
