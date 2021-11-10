import cv2 as cv
from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from matplotlib.pyplot import imsave, imread
import matplotlib.pyplot as plt
import os
import glob


"""
Takes in a single file from the InputImages folder and creates a single file output in the Outputs folder.

Processing takes any almost black pixels from the original and draws the coordinates on top of this canvas.
I do this as some input images already have a black outline around some parts of the image. The result is then that CED creates a cavity edge
around this black outline and it looks unpleasant.

This file uses:
cluster size 7 which is chosen arbitrarily.
kernel_size which is calculated from the number of unique colours. The logic being that an image with lots of unique colours
will require more blurring, and hence a bigger kernel size is preferred.

The image is split into an n x n grid and the kernel_size is determined for each specific section, blurred, and then recombined for the remaining
processing.

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

    @staticmethod
    def calculate_filter_size(image):
        # We calculate the number of unique colours to determine the kernel_size. An image with a lot of unique
        # colours requires more blurring, which is achieved with a bigger kernel_size.
        unique_colors = set()
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                pixel = image.getpixel((i, j))
                unique_colors.add(pixel)

        filter_size = int(str(round(len(unique_colors), -3))[0])

        # Filter size needs to be odd. Sometimes I forget and put an even filter size. This will catch this error and
        # reduce it by one to make it odd again.
        if filter_size % 2 == 0:
            filter_size = max(5, filter_size + 1)

        else:
            filter_size = max(5, filter_size)

        return filter_size

    def filter_over_section(self):
        import image_slicer
        from image_slicer import join
        # Cut a section of the image
        im = np.array(self.imag)
        imsave(f'D:\GitHub\ImprovedColouringBook\FlaskApp\static\img\Tile.jpg', im)

        num_tiles = 9
        tiles = image_slicer.slice("D:\GitHub\ImprovedColouringBook\FlaskApp\static\img\Tile.jpg", num_tiles)
        tile_index = 0
        self.filter_sizes = []
        for tile in tiles:
            # Read in image as array
            temp_tile = imread(tile.filename)

            # calculate_filter_size requires an Image format, as get_pixel does not work on arrays. We need to save the temp_tile and reopen as an
            # image.
            imsave(f'D:\GitHub\ImprovedColouringBook\FlaskApp\static\img\{tile_index}.jpg', temp_tile)
            temp_tile = Image.open(f'D:\GitHub\ImprovedColouringBook\FlaskApp\static\img\{tile_index}.jpg')

            # Calculate filter_size for this particular tile
            filter_size = self.calculate_filter_size(temp_tile)
            self.filter_sizes.append((tile_index, filter_size))

            tile_index += 1

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

        # Implement k-means clustering to form k clusters
        kmeans = MiniBatchKMeans(n_clusters=self.clusters)
        kmeans.fit(imag)

        # Replace each pixel value with its nearby centroid
        compressed_image = kmeans.cluster_centers_[kmeans.labels_]
        compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

        # Reshape the image to original dimension
        self.compressed_image = compressed_image.reshape(rows, cols, 3)


    def median_over_section(self):
        import image_slicer
        from image_slicer import join
        # Save the kmeans image as an array
        full_kmeans = Image.fromarray(self.compressed_image)

        # image_slicer wants a string input for the file so we must save this first and then reopen it.
        imsave(f'D:\GitHub\ImprovedColouringBook\FlaskApp\static\img\{self.file_name}kmeans.jpg', full_kmeans)

        num_tiles = 9
        tiles = image_slicer.slice(f'D:\GitHub\ImprovedColouringBook\FlaskApp\static\img\{self.file_name}kmeans.jpg', num_tiles)

        tile_index = 0
        for tile in tiles:
            # Read in image as array
            temp_tile = imread(tile.filename)
            print(self.filter_sizes[tile_index][1])
            median_tile = cv.medianBlur(temp_tile, self.filter_sizes[tile_index][1])
            #plt.imshow(median_tile, cmap='gray')
            #plt.show()
            tile_index += 1


    def auto_canny(self, sigma=0.33):
        self.canny = cv.cvtColor(self.compressed_image, cv.COLOR_RGB2GRAY)

        # compute the median of the single channel pixel intensities
        v = np.median(self.canny)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (0.3 - sigma) * v))
        upper = int(min(255, (0.8 - sigma) * v))

        self.edged = cv.Canny(self.canny, lower, upper)


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


file_name = "hulk"

pic = Canny(name=f'{file_name}.jpg', clusters=11, scalar=1)
pic.filter_over_section()
pic.k_means()
pic.median_over_section()
pic.auto_canny()
pic.draw_contours()
