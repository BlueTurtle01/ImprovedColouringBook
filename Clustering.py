#Credit: https://towardsdatascience.com/image-compression-using-k-means-clustering-aa0c91bb0eeb
# Credit: https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_imgproc/py_filtering/py_filtering.html
from matplotlib.pyplot import imread, imshow, imsave
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

file_name = "family"
file_path = str("InputImages/" + file_name + ".jpg")

def kmeans(k, file_path = file_path):
    #Read the image
    image = imread(file_path)
    imshow(image)

    #Dimension of the original image
    rows = image.shape[0]
    cols = image.shape[1]

    #Flatten the image
    image = image.reshape(rows*cols, 3)

    #Implement k-means clustering to form k clusters
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)

    #Replace each pixel value with its nearby centroid
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

    #Reshape the image to original dimension
    compressed_image = compressed_image.reshape(rows, cols, 3)

    #Save and display output image
    imsave(("OutputImages/" + str(file_name) + str(k) + "Clustered.jpg"), compressed_image)
    #imshow(compressed_image)
    #plt.show()

    return compressed_image
