from matplotlib.pyplot import imsave
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
import glob
from UtilityFunctions import directory_creator
import os
from distinctipy import distinctipy
import cProfile
import pstats


class Canny:
    def __init__(self, name, path, clusters, output, scalar):
        self.file_name = name
        self.file_path = path
        self.clusters = clusters
        self.output_path = output
        self.imag = Image.open(self.file_path)
        self.height, self.width = self.imag.size
        self.scalar = scalar

    def k_means(self):
        #Resize the image in the hopes that kmeans and contours can find the edges easier.
        imag = self.imag.resize([int(self.scalar * s) for s in self.imag.size], Image.ANTIALIAS)

        unique_colors = set()
        for i in range(imag.size[0]):
            for j in range(imag.size[1]):
                pixel = imag.getpixel((i, j))
                unique_colors.add(pixel)

        print('Unique color count = ', len(unique_colors))

        self.filter_size = int(str(round(len(unique_colors), -3))[0])

        # Dimension of the original image
        cols, rows = imag.size

        # Flatten the image with the new dimensions.
        imag = np.array(imag).reshape(rows * cols, 3)

        # Implement k-means clustering to form k clusters
        #kmeans = KMeans(n_clusters=self.clusters)
        kmeans = MiniBatchKMeans(n_clusters=self.clusters)
        kmeans.fit(imag)

        # Replace each pixel value with its nearby centroid
        compressed_image = kmeans.cluster_centers_[kmeans.labels_]
        compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

        # Reshape the image to original dimension
        self.compressed_image = compressed_image.reshape(rows, cols, 3)

        # Save and display output image
        plt.imshow(self.compressed_image)
        plt.title("Original kMeans")
        imsave((str(self.output_path) + str(self.file_name) + "kMeans.png"), self.compressed_image)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def median(self, filter_size=7):

        # Filter size needs to be odd. Sometimes I forget and put an even filter size. This will catch this error and
        # reduce it by one to make it odd again.
        if self.filter_size % 2 == 0:
            self.filter_size = max(5, self.filter_size + 1)

        else:
            self.filter_size = max(5, self.filter_size)

        from skimage.restoration import estimate_sigma

        #noise = estimate_sigma(self.compressed_image, multichannel=True, average_sigmas=True)

        self.compressed_image = cv2.resize(self.compressed_image,
                                           dsize=(self.height * self.scalar, self.width * self.scalar),
                                           interpolation=cv2.INTER_AREA)

        self.median = cv2.medianBlur(self.compressed_image, self.filter_size)
        plt.imshow(self.median)
        plt.title("Median Blur")
        plt.xticks([])
        plt.yticks([])
        imsave((str(self.output_path) + "MedianBlur.png"), self.median)
        plt.show()

        median_gray = cv2.cvtColor(self.median, cv2.COLOR_RGB2GRAY)
        imsave((str(self.output_path) + "MedianBlurGray.png"), median_gray, cmap='gray')


    def replace_colours(self, clusters):
        unique, counts = np.unique(self.median.reshape(-1, 3), axis=0, return_counts=True)

        colors = distinctipy.get_colors(clusters)

        colour_list = []
        for item in colors:
            red, green, blue = item
            colour = (int(red * 255), int(green * 255), int(blue * 255))

            colour_list.append(colour)

        index = 0
        for colour in unique:
            lower = np.array([colour[0], colour[1], colour[2]])
            try:
                #mask = cv2.inRange(self.median, lower, lower)

                #self.median[mask > 0] = colour_list[index]

                self.median = np.where(self.median == lower, colour_list[index], self.median)
                index += 1
            except IndexError:
                pass

        plt.imshow(self.median)
        plt.title("Removed")
        plt.xticks([])
        plt.yticks([])
        imsave((str(self.output_path) + "DistinctColoured.png"), self.median.astype('uint8'))
        plt.show()


    def auto_canny(self, sigma=0.33):
        self.canny = cv2.cvtColor(self.median, cv2.COLOR_RGB2GRAY)

        # compute the median of the single channel pixel intensities
        v = np.median(self.canny)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (0.3 - sigma) * v))
        upper = int(min(255, (0.8 - sigma) * v))

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

        plt.imshow(contours_, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.title("Contours")
        imsave((str(self.output_path) + "Contours.png"), contours_, cmap="gray")
        plt.show()


file_names = [os.path.basename(x) for x in glob.glob('MultiInputs/*.jpg')]

for file_name in file_names:
    file_name = file_name.split(".", 2)[0]
    file_path = str("MultiInputs/" + file_name + ".jpg")
    directory_creator(file_name=file_name)

    output_path = ("MultiOutput/" + file_name + "/")

    cr = cProfile.Profile()
    cr.enable()
    pic = Canny(name=file_name, path=file_path, clusters=7, output=("MultiOutput/" + file_name + "/"), scalar=1)
    pic.k_means()
    pic.median()
    #pic.replace_colours(clusters)
    pic.auto_canny(sigma=0.33)
    pic.draw_contours()
    cr.disable()
    stats = pstats.Stats(cr).sort_stats('tottime')
    stats.print_stats(15)
