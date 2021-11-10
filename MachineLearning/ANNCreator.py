import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import random
import pandas as pd
from matplotlib.pyplot import imsave
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from PythonScripts.UtilityFunctions import directory_creator
from csv import writer
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 3


"""
File creates a dataframe if none exists, and then runs multiple configurations of the input image. The output images are saved to their own folder and
the configuration settings are saved to a csv file.

This file is a precursor to the SelectRandomFile.py file which then picks a random configuration and asks the user their favourite.
"""


def create_dataframe():
    # Create a blank dataframe
    data = pd.DataFrame(columns=['ImageID', 'Clusters', 'Filter size', 'Sigma', 'Unique Colours', 'Chosen', 'Pixels']).set_index('ImageID')

    # Save it to a CSV
    if not os.path.isfile('Values.csv'):
        data.to_csv('Values.csv')


class Canny:
    def __init__(self, name, path, clusters, output, scalar, sigma=0.33):
        self.file_name = name
        self.file_path = path
        self.clusters = clusters
        self.output_path = output
        self.imag = Image.open(self.file_path)
        self.height, self.width = self.imag.size
        self.scalar = scalar
        self.sigma = sigma
        self.chosen = 0

    def k_means(self):
        #Resize the image in the hopes that kmeans and contours can find the edges easier.
        imag = self.imag.resize([int(self.scalar * s) for s in self.imag.size], Image.ANTIALIAS)
        print(type(imag))

        unique_colors = set()
        for i in range(imag.size[0]):
            for j in range(imag.size[1]):
                pixel = imag.getpixel((i, j))
                unique_colors.add(pixel)

        numb_colours = len(unique_colors)
        self.filter_size = int(str(round(len(unique_colors), -(len(str(numb_colours))-2)))[0:(len(str(numb_colours))-2)-2])
        self.unique_colours = len(unique_colors)

        # Dimension of the original image
        self.cols, self.rows = imag.size

        # Flatten the image with the new dimensions.
        imag = np.array(imag).reshape(self.rows * self.cols, 3)

        # Implement k-means clustering to form k clusters
        kmeans = MiniBatchKMeans(n_clusters=self.clusters)
        kmeans.fit(imag)

        # Replace each pixel value with its nearby centroid
        compressed_image = kmeans.cluster_centers_[kmeans.labels_]
        compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

        # Reshape the image to original dimension
        self.compressed_image = compressed_image.reshape(self.rows, self.cols, 3)

    def median(self):

        # Filter size needs to be odd. Sometimes I forget and put an even filter size. This will catch this error and
        # reduce it by one to make it odd again.
        if self.filter_size % 2 == 0:
            self.filter_size = max(7, self.filter_size + 1)

        else:
            self.filter_size = max(7, self.filter_size)

        self.compressed_image = cv2.resize(self.compressed_image,
                                           dsize=(self.height * self.scalar, self.width * self.scalar),
                                           interpolation=cv2.INTER_AREA)

        self.median = cv2.medianBlur(self.compressed_image, self.filter_size)


    def auto_canny(self):
        self.canny = cv2.cvtColor(self.median, cv2.COLOR_RGB2GRAY)

        # compute the median of the single channel pixel intensities
        v = np.median(self.canny)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (0.3 - self.sigma) * v))
        upper = int(min(255, (0.8 - self.sigma) * v))

        self.edged = cv2.Canny(self.canny, lower, upper)


    def draw_contours(self):
        """
        Draw the contours over a blank array. The function cv2.DrawContours overlays the contours on top of the bitwise array.
        Which is not ideal if the bitwise array contains some small, noisy contours. Therefore, I created an empty array first and then used this as the base
        for drawing the contours onto.
        :param edged: Output of the Canny Edge Detection algorithm after applying erode and dilation.
        :return:
        """
        contours, hierarchy = cv2.findContours(self.edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

        edged = cv2.bitwise_not(self.edged)
        rgb = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)

        temp_array = np.ones([rgb.shape[0], rgb.shape[1], rgb.shape[2]])
        contours_ = cv2.drawContours(temp_array, contours, -1, (0, 0, 0), thickness=1)

        ml_filename = 'MLOutputs/' + str(self.file_name) + 'Clusters' + str(self.clusters) + \
                      'FilterSize' + str(self.filter_size) + 'Sigma' + str(self.sigma) + 'UniqueColours' + \
                      str(self.unique_colours) + ".png"

        plt.imshow(contours_, cmap="gray")
        if not os.path.isfile(ml_filename):
            imsave(ml_filename, contours_, cmap="gray")


    def append_list_as_row(self):
        # Open file in append mode
        with open('Values.csv', 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow([self.file_name, self.clusters, self.filter_size, self.sigma, self.unique_colours, self.chosen, (self.cols*self.rows)])


create_dataframe()
file_names = [os.path.basename(x) for x in glob.glob('MLInputs/*.jpg')]
#random_file_id = random.randint(0, (len(file_names)-1))
#file_name = file_names[random_file_id]
numb_outputs = 4

# For each file in the relevant folder - in this case MultiInputs
for file in range(len(file_names)):
    # Deconstruct the filename by removing the extension and creating a directory for the file.
    # In this directory we will keep all permutations of the output image that occur from the various kernel
    # sizes and number of clusters.
    file_name = file_names[file]
    file_name = file_name.split(".", 2)[0]
    file_path = f'MLInputs/{file_name}.jpg'
    directory_creator(file_name=file_name)

    output_path = f'MLOutputs/{file_name}/'

    cluster_values = set()
    sigma_values = set()

    while len(cluster_values) < numb_outputs:
        clusters = random.randint(5, 20)

        # So we don't repeat the computation for this image for a previous computed cluster value, we keep a track of this using
        # the blank set created above. Once a cluster value is used it is added to this set.
        if clusters not in cluster_values:

            # Similarly for sigma values to avoid repetition.
            sigma = round(random.uniform(0.2, 0.5), 2)
            if sigma not in sigma_values:
                pic = Canny(name=file_name, path=file_path, clusters=clusters, output=f'MLOutputs/{file_name}/',
                            scalar=1, sigma=sigma)
                pic.k_means()
                pic.median()
                pic.auto_canny()
                pic.draw_contours()
                pic.append_list_as_row()
                cluster_values.add(clusters)
                sigma_values.add(sigma)


