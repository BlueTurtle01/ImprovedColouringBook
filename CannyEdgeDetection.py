from matplotlib.pyplot import imread, imshow, imsave
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.cluster import KMeans


class Canny:
    def __init__(self, name, path, clusters, output):
        self.file_name = name
        self.file_path = path
        self.clusters = clusters
        self.output_path = output

    def k_means(self):
        # Read the image
        imag = Image.open(self.file_path)
        imshow(imag)

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
        #imsave(("OutputImages/" + str(self.file_name) + str(self.clusters) + "Clustered.jpg"), self.compressed_image)

    def canny_edge_detection(self):
        self.img = cv2.cvtColor(self.compressed_image, cv2.COLOR_BGR2GRAY)

    def gaussian_filter(self, k):
        self.blurred = cv2.GaussianBlur(self.img, (k, k), sigmaX=(k - 1) / 6)
        plt.imshow(self.blurred)
        plt.show()

    def bilateral_blur(self, k):
        self.blurred = cv2.bilateralFilter(self.img, k, 75, 75)
        plt.imshow(self.blurred)
        plt.show()

    def median(self, filter_size=7):
        self.blurred = cv2.medianBlur(self.img, filter_size)
        plt.imshow(self.blurred)
        plt.show()

    def auto_canny(self, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(self.blurred)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        self.edged = cv2.Canny(self.blurred, lower, upper)

    def canny(self):
        rgb = cv2.cvtColor(self.edged, cv2.COLOR_GRAY2RGB)
        rgb = cv2.dilate(rgb, (20, 20), iterations=1)
        rgb = cv2.erode(rgb, (5, 5), iterations=1)
        self.rgb = cv2.bitwise_not(rgb)  # Credit: https://stackoverflow.com/a/40954142/4367851

        plt.imshow(self.rgb)
        plt.title("Canny")
        plt.xticks([])
        plt.yticks([])
        imsave((str(self.output_path) + "Canny.jpg"), self.rgb, cmap="gray")
        plt.show()

    def draw_contours(self):
        """
        Draw the contours over a blank array. The function cv2.DrawContours overlays the contours on top of the bitwise array.
        Which is not ideal if the bitwise array contains some small, noisy contours. Therefore, I created an empty array first and then used this as the base
        for drawing the contours onto.
        :param edged: Output of the Canny Edge Detection algorithm after applying erode and dilation.
        :return:
        """
        contours, hierarchy = cv2.findContours(self.edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        edged = cv2.bitwise_not(self.edged)
        rgb = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB)
        print("Number of Contours found = " + str(len(contours)))

        final_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 0:
                final_contours.append(contour)
            else:
                pass

        temp_array = np.ones([rgb.shape[0], rgb.shape[1], rgb.shape[2]])
        contours_ = cv2.drawContours(temp_array, final_contours, -1, (0, 0, 0), thickness=-1)

        plt.imshow(contours_, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.title("Contours")
        plt.show()
        imsave((str(self.output_path) + "Contours.jpg"), contours_, cmap="gray")


file_name = "donald"
pic = Canny(name=file_name, path=("InputImages/" + file_name + ".jpg"), clusters=10, output=("OutputImages/" + file_name + "/"))
pic.k_means()
pic.canny_edge_detection()
'''pic.gaussian_filter(k=7)'''
pic.bilateral_blur(k=7)
pic.median(filter_size=5)
pic.auto_canny(sigma=0.33)
pic.canny()
pic.draw_contours()
