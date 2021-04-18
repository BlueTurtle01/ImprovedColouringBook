from matplotlib.pyplot import imsave
import numpy as np
import cv2
from sklearn.cluster import KMeans
from time import perf_counter
from kneed import KneeLocator
from Plots import plot_elbow_method, plot_histogram, plot_image
from UtilityFunctions import directory_creator


file_name = "ash"
file_path = str("InputImages/" + file_name + ".jpg")
output_path = ("OutputImages/" + str(file_name) + "/")
# TODO: Change this to multi input


class CreateContour:
    def __init__(self, file_name, file_path, output_path):
        self.file_name = file_name
        self.file_path = file_path
        self.output_path = output_path

    def calculate_clusters(self):
        # Credit: https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
        original_image = cv2.imread(self.file_path)

        # Create a directory for the outputs if one does not already exist
        directory_creator(file_name=self.file_name)

        imsave("OutputImages/" + str(self.file_name) + "/Original Image.jpg", original_image)

        # Save the original image in RGB
        plot_image(image=cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), output_path=self.output_path, title="Original Image")
        # plot_histogram(original_image)

        Sum_of_squared_distances = []
        times = []
        # Dimension of the original image
        rows, cols, _ = original_image.shape

        # Flatten the image
        imag = np.array(original_image).reshape(rows * cols, 3)

        K = range(5, 31, 5)
        for k in K:
            start = perf_counter()
            print(k)
            km = KMeans(n_clusters=k)
            km.fit(imag)
            stop = perf_counter()
            times.append((stop - start))
            Sum_of_squared_distances.append(km.inertia_)

            #  Replace each pixel value with its nearby centroid
            compressed_image = km.cluster_centers_[km.labels_]
            compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

            #  Reshape the image to original dimension
            compressed_image = compressed_image.reshape(rows, cols, 3)
            compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)

            imsave((str(self.output_path) + str(k) + "Clustered.jpg"), compressed_image)

        kneedle = KneeLocator(K, Sum_of_squared_distances, S=1.0, curve="convex", direction="decreasing")

        # plot_elbow_method: arguments (x, y1, y2, elbow point)
        plot_elbow_method(Sum_of_squared_distances, times, K, knee=kneedle.knee, output_path=self.output_path)

        # Implement k-means clustering to form k clusters
        kmeans = KMeans(n_clusters=kneedle.knee)
        kmeans.fit(imag)

        #  Replace each pixel value with its nearby centroid
        compressed_image = kmeans.cluster_centers_[kmeans.labels_]
        compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

        #  Reshape the image to original dimension
        self.compressed_image = compressed_image.reshape(rows, cols, 3)

        return kneedle.knee, original_image

    def auto_canny(self, sigma=0.5):
        # compute the median of the single channel pixel intensities
        v = np.median(self.compressed_image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        self.edged = cv2.Canny(self.compressed_image, lower, upper)

    def morpho_transforms(self):
        edged = cv2.dilate(self.edged, (20, 20), iterations=1)
        self.edged = cv2.erode(edged, (20, 20), iterations=1)

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
            if cv2.contourArea(contour) > 40:
                final_contours.append(contour)
                pass
            else:
                pass

        temp_array = np.ones([rgb.shape[0], rgb.shape[1], rgb.shape[2]])
        contours_ = cv2.drawContours(temp_array, final_contours, -1, (0, 0, 0), thickness=3)

        plot_image(image=contours_, output_path=output_path, title="Contours")


Contour = CreateContour(file_name=file_name, file_path=file_path, output_path=output_path)
Contour.calculate_clusters()
Contour.auto_canny()
Contour.morpho_transforms()
Contour.draw_contours()

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
