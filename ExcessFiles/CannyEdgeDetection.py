from matplotlib.pyplot import imread, imshow, imsave
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import glob
from UtilityFunctions import directory_creator
import os
from distinctipy import distinctipy
import sys


class Canny:
    def __init__(self, name, path, clusters, output):
        self.file_name = name
        self.file_path = path
        self.clusters = clusters
        self.output_path = output

    def k_means(self):
        # Read the image
        imag = Image.open(self.file_path)

        imag = imag.resize([int(1 * s) for s in imag.size], Image.ANTIALIAS)

        # Dimension of the original image
        cols, rows = imag.size

        # Flatten the image
        imag = np.array(imag).reshape(rows * cols, 3)

        # Implement k-means clustering to form k clusters
        kmeans = KMeans(n_clusters=self.clusters)
        kmeans.fit(imag)

        # Replace each pixel value with its nearby centroid
        compressed_image = kmeans.cluster_centers_[kmeans.labels_]
        compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

        # Reshape the image to original dimension
        self.compressed_image = compressed_image.reshape(rows, cols, 3)

        # Save and display output image
        plt.imshow(self.compressed_image)
        plt.title("Original kMeans")
        imsave((str(self.output_path) + str(self.file_name) + "kMeans.jpg"), self.compressed_image)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def dom_colours(self):
        # Create a copy of the image so as to not break the original
        img_temp = self.compressed_image.copy()

        # Find the size of the image
        height, width = img_temp.shape[0], img_temp.shape[1]
        blank_array = np.zeros([height, width, 3])

        # Define interval size. This is a square of n x n. The larger the interval the more simplification will occur.
        interval_size = 5

        # Using the interval size and the image sizes, calculate the number of rows and columns.
        rows = int(height / interval_size)
        cols = int(width / interval_size)

        # For each row and column...
        for row in range(rows):
            for col in range(cols):
                # Read in the interval we are currently concerned with. This will be a square of n x n.
                result = img_temp[row * interval_size: row * interval_size + interval_size, col * interval_size: col * interval_size + interval_size]

                # Find the unique colour values in the result array and count the occurrences of each.
                unique, counts = np.unique(result.reshape(-1, 3), axis=0, return_counts=True)

                blank_array[row * interval_size: row * interval_size + interval_size, col * interval_size: col * interval_size + interval_size, 0] = unique[np.argmax(counts)][0]
                blank_array[row * interval_size: row * interval_size + interval_size, col * interval_size: col * interval_size + interval_size, 1] = unique[np.argmax(counts)][1]
                blank_array[row * interval_size: row * interval_size + interval_size, col * interval_size: col * interval_size + interval_size, 2] = unique[np.argmax(counts)][2]
        plt.imshow(blank_array.astype('uint8'))
        plt.title("Smoothed kMeans")
        imsave((str(self.output_path) + "Smoothed.jpg"), blank_array.astype('uint8'))
        plt.xticks([])
        plt.yticks([])
        plt.show()
        self.compressed_image = blank_array.astype('uint8')


    def replace_colours(self):
        unique, counts = np.unique(self.compressed_image.reshape(-1, 3), axis=0, return_counts=True)

        colors = distinctipy.get_colors(20)
        distinctipy.color_swatch(colors)


        colour_list = []
        for item in colors:
            red, green, blue = item
            colour = (int(red * 255), int(green * 255), int(blue * 255))

            colour_list.append(colour)

        self.copied_compressed = self.compressed_image.copy()
        index = 0
        for colour in unique:
            lower = np.array([colour[0], colour[1], colour[2]])
            # Mask image to only select browns
            try:
                mask = cv2.inRange(self.copied_compressed, lower, lower)

                self.copied_compressed[mask > 0] = colour_list[index]
                index += 1
            except IndexError:
                pass

        plt.imshow(self.copied_compressed)
        plt.title("Removed")
        plt.xticks([])
        plt.yticks([])
        imsave((str(self.output_path) + "DistinctColoured.jpg"), self.copied_compressed, cmap="viridis")
        plt.show()


    def median(self, filter_size=7):
        self.median = cv2.medianBlur(self.copied_compressed, filter_size)
        self.median_orig = cv2.medianBlur(self.compressed_image, filter_size)
        plt.imshow(self.compressed_image)
        plt.title("Median Blur")
        plt.xticks([])
        plt.yticks([])
        imsave((str(self.output_path) + "MedianBlur.jpg"), self.median_orig)
        plt.show()


    def canny_edge_detection(self):
        self.canny = cv2.cvtColor(self.median, cv2.COLOR_RGB2GRAY)


    def auto_canny(self, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(self.canny)
        print(v)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (0.5 - sigma) * v))
        print("Lower", lower)

        upper = int(min(255, (0.8 - sigma) * v))
        print("Upper", upper)

        self.edged = cv2.Canny(self.canny, lower, upper)


    def draw_contours(self):
        """
        Draw the contours over a blank array. The function cv2.DrawContours overlays the contours on top of the bitwise array.
        Which is not ideal if the bitwise array contains some small, noisy contours. Therefore, I created an empty array first and then used this as the base
        for drawing the contours onto.
        :param edged: Output of the Canny Edge Detection algorithm after applying erode and dilation.
        :return:
        """
        contours, hierarchy = cv2.findContours(self.edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

        edged = cv2.bitwise_not(self.edged)
        rgb = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
        #print("Number of Contours found = " + str(len(contours)))

        final_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 20:
                final_contours.append(contour)
            else:
                pass

        temp_array = np.ones([rgb.shape[0], rgb.shape[1], rgb.shape[2]])
        contours_ = cv2.drawContours(temp_array, contours, -1, (0, 0, 0), thickness=1)

        plt.imshow(contours_, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.title("Contours")
        imsave((str(self.output_path) + "Contours1.png"), contours_, cmap="gray")
        plt.show()


file_names = [os.path.basename(x) for x in glob.glob('MultiInputs/*.jpg')]

for file_name in file_names:
    file_name = file_name.split(".", 2)[0]
    file_path = str("MultiInputs/" + file_name + ".jpg")
    directory_creator(file_name=file_name)

    output_path = ("MultiOutput/" + file_name + "/")

    pic = Canny(name=file_name, path=file_path, clusters=15, output=("MultiOutput/" + file_name + "/"))
    pic.k_means()

    pic.replace_colours()
    pic.median(filter_size=13)
    pic.canny_edge_detection()
    pic.auto_canny(sigma=0.20)
    pic.draw_contours()
