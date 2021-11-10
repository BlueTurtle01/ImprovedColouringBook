import os
os.environ["OMP_NUM_THREADS"] = "1"
import cv2 as cv
from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from Plots import plot_elbow_method
from kneed import KneeLocator
from time import perf_counter



"""
Takes in a single file from the InputImages folder and creates a single file output in the Outputs folder.

I implement the Elbow method here to calculate the optimal number of clusters.
"""


class Canny:
    def __init__(self, name, clusters, scalar):
        self.file_name = name
        self.imag = Image.open(f'InputImages/{self.file_name}')
        self.gray = cv.cvtColor(np.array(self.imag), cv.COLOR_BGR2GRAY)
        thresh, self.bw = cv.threshold(self.gray, 20, 255, cv.THRESH_BINARY)
        self.clusters = clusters
        self.height, self.width = self.imag.size
        self.scalar = scalar

    def calculate_filter_size(self):
        # We calculate the number of unique colours to determine the kernel_size. An image with a lot of unique
        # colours requires more blurring, which is achieved with a bigger kernel_size.
        unique_colors = set()
        for i in range(self.imag.size[0]):
            for j in range(self.imag.size[1]):
                pixel = self.imag.getpixel((i, j))
                unique_colors.add(pixel)

        filter_size = int(str(round(len(unique_colors), -3))[0])

        # Filter size needs to be odd. Sometimes I forget and put an even filter size. This will catch this error and
        # reduce it by one to make it odd again.
        if filter_size % 2 == 0:
            filter_size = max(5, filter_size + 1)

        else:
            filter_size = max(5, filter_size)

        return filter_size

    def k_means(self):
        # Resize the image in the hopes that kmeans and contours can find the edges easier.
        # This also reduces computational load.
        w, h = self.imag.size
        if w > 1000:
            h = int(h * 1000. / w)
            w = 1000
        imag = self.imag.resize((w, h), Image.NEAREST)

        # Dimension of the original image
        cols, rows = imag.size

        # Flatten the image with the new dimensions.
        imag = np.array(imag).reshape(rows * cols, 3)

        Sum_of_squared_distances = []
        times = []

        K = range(5, 21, 1)
        for k in K:
            print(k)
            start = perf_counter()
            km = MiniBatchKMeans(n_clusters=k)
            km.fit(imag)
            stop = perf_counter()
            times.append(stop - start)
            Sum_of_squared_distances.append(km.inertia_)

        kneedle = KneeLocator(K, Sum_of_squared_distances, S=1.0, curve="convex", direction="decreasing")
        plot_elbow_method(Sum_of_squared_distances, times, K, knee=kneedle.knee)

        # Implement k-means clustering to form k clusters
        kmeans = MiniBatchKMeans(n_clusters=kneedle.knee)
        kmeans.fit(imag)

        # Replace each pixel value with its nearby centroid
        compressed_image = kmeans.cluster_centers_[kmeans.labels_]
        compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

        # Reshape the image to original dimension
        self.compressed_image = compressed_image.reshape(rows, cols, 3)
        print(type(self.compressed_image))


    def median(self):

        self.compressed_image = cv.resize(self.compressed_image,
                                          dsize=(self.height * self.scalar, self.width * self.scalar),
                                          interpolation=cv.INTER_NEAREST)

        filter_size = self.calculate_filter_size()
        self.median = cv.medianBlur(self.compressed_image, filter_size)

    def auto_canny(self, sigma=0.33):
        self.canny = cv.cvtColor(self.median, cv.COLOR_RGB2GRAY)

        # compute the median of the single channel pixel intensities
        v = np.median(self.canny)
        # apply automatic Canny edge detection using the computed median
        #lower = int(max(0, (0.3 - sigma) * v))
        #upper = int(min(255, (0.8 - sigma) * v))
        upper = 100
        lower = 5

        self.edged = cv.Canny(self.canny, lower, upper, L2gradient=True)


    def draw_contours(self):
        contours, hierarchy = cv.findContours(self.edged, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)

        with open(f'{self.file_name.split(".")[0]}.svg', "w+") as f:
            f.write(f'<svg width="{self.height}px" height="{self.width}px" xmlns="http://www.w3.org/2000/svg">')

            for c in contours:
                f.write('<path d="M')
                for i in range(len(c)):
                    x, y = c[i][0]
                    f.write(f"{x} {y} ")
                f.write('" style="stroke:black;fill:none"/>')
            f.write("</svg>")


file_name = "TheSimpsons_ZABF22_PodcastNews_Sc2010AvidColorCorrected-e1605545531109"

pic = Canny(name=f'{file_name}.jpg', clusters=7, scalar=1)
pic.k_means()
pic.median()
pic.auto_canny()
pic.draw_contours()
