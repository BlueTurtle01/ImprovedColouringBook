from matplotlib.pyplot import imsave
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans
import pandas as pd
import os
from PIL import Image
import glob


class Colouring:
    def __init__(self, name, output, path):
        self.file_name = name
        self.output_path = output
        self.file_path = path
        self.original_image = cv2.imread(self.file_path)

    def calculate_optimal_clusters(self):
        sum_of_squared_distances = []
        # Dimension of the original image
        rows, cols, _ = self.original_image.shape

        # Flatten the image
        imag = np.array(self.original_image).reshape(rows * cols, 3)

        K = range(10, 11)
        for k in K:
            km = KMeans(n_clusters=k)
            km.fit(imag)
            sum_of_squared_distances.append(km.inertia_)

        error_df = pd.DataFrame(list(zip(K, sum_of_squared_distances)), columns=["K", "Error"])
        error_df["Difference"] = error_df["Error"] - error_df["Error"].shift(-1)

        try:
            self.clusters = min(error_df.index[error_df["Difference"] < 3e+06].tolist())
        except ValueError:
            self.clusters = max(error_df["K"])

        # Implement k-means clustering to form k clusters
        k_means = KMeans(n_clusters=self.clusters)
        k_means.fit(imag)

        #  Replace each pixel value with its nearby centroid
        self.compressed_image = k_means.cluster_centers_[k_means.labels_]
        self.compressed_image = np.clip(self.compressed_image.astype('uint8'), 0, 255)

        #  Reshape the image to original dimension
        self.compressed_image = self.compressed_image.reshape(rows, cols, 3)

        imsave((self.output_path + self.file_name + str(self.clusters) + "Clustered.jpg"), self.compressed_image)

    def thresholding(self, k):
        # The output of the k-means algorithm is an array in the form of Blue,Green,Red (BGR) which needs to be converted to grayscale before we conduct a Gaussian Blur.
        self.compressed_img = cv2.cvtColor(self.compressed_image, cv2.COLOR_BGR2GRAY)  # imread reads images as Blue, Green, Red, Alpha

        # Size of the kernel for the Gaussian Blur
        blurred = cv2.GaussianBlur(self.compressed_img, (k, k), sigmaX=(k - 1) / 6)

        # Conduct Adaptive Thresholding to the blurred image
        threshold_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 3)
        threshold_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, (15, 15))
        imsave((self.output_path + "Threshold Gray.jpg"), threshold_image, cmap="gray")


file_names = [os.path.basename(x) for x in glob.glob('MultiInputs/*.jpg')]

for file_name in file_names:
    file_name = file_name.split(".", 2)[0]
    file_path = str("MultiInputs/" + file_name + ".jpg")
    output_path = ("MultiOutput/" + file_name + "/")

    def create_directory():
        # Create a directory for the outputs if one does not already exist
        try:
            os.mkdir(str(output_path))
        except FileExistsError:
            pass

    create_directory()

    colouring = Colouring(name=file_name, output=output_path, path=file_path)
    #colouring.plot_image()
    colouring.calculate_optimal_clusters()
    colouring.thresholding(k=9)  # k: kernel size for Gaussian Blur


