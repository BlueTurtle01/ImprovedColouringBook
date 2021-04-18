from matplotlib.pyplot import imsave
from sklearn.cluster import KMeans
import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
from PIL import Image, ImageColor, ImageDraw, ImageFont
import cv2
from kneed import KneeLocator
import glob
from UtilityFunctions import directory_creator
from Plots import plot_elbow_method
from time import perf_counter
import warnings
warnings.filterwarnings('ignore')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
    A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin

    return left, right, top, bottom


class RegionDetector:
    def __init__(self, file_name, file_path, path):
        self.file_name = file_name
        self.file_path = file_path
        self.output_path = path


    def load_img(self):
        self.img = cv2.imread(self.file_path)
        self.width, self.height, _ = self.img.shape

    def load_model(self):
        model = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
        detector = hub.load(model).signatures['default']
        converted_img = tf.image.convert_image_dtype(self.img, tf.float32)[tf.newaxis, ...]
        result = detector(converted_img)
        self.result = {key: value.numpy() for key, value in result.items()}

    def draw_boxes(self, max_boxes=2, min_score=0.5):
        boxes = self.result["detection_boxes"]
        class_names = self.result["detection_class_entities"]
        scores = self.result["detection_scores"]

        """Overlay labeled boxes on an image with formatted scores and label names."""
        colors = list(ImageColor.colormap.values())
        font = ImageFont.load_default()

        for i in range(min(boxes.shape[0], max_boxes)):
            if scores[i] >= min_score:
                ymin, xmin, ymax, xmax = tuple(boxes[i])
                display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
                color = colors[hash(class_names[i]) % len(colors)]
                image_ = Image.fromarray(np.uint8(self.img)).convert("RGB")
                left, right, top, bottom = draw_bounding_box_on_image(image_, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
                if i == 0:
                    large_box = np.zeros([1, 4])
                    large_box[0, 0] = left
                    large_box[0, 1] = right
                    large_box[0, 2] = top
                    large_box[0, 3] = bottom
                else:
                    pass
                np.copyto(self.img, np.array(image_))
                large_box[0, 0] = min(large_box[0, 0], left)
                large_box[0, 1] = max(large_box[0, 1], right)
                large_box[0, 2] = min(large_box[0, 2], top)
                large_box[0, 3] = max(large_box[0, 3], bottom)

        left, right, top, bottom = large_box.T
        image_ = cv2.cvtColor(np.array(image_), cv2.COLOR_BGR2RGB)
        plt.imshow(image_)
        plt.show()

        return left, right, top, bottom


# GrabCut with regions
class GrabCut:
    def __init__(self, name, file_path, path):
        self.file_name = name
        self.output_path = path
        self.file_path = file_path
        self.original_image = cv2.imread(self.file_path)

    def create_directory(self):
        # Create a directory for the outputs if one does not already exist
        try:
            os.mkdir(self.output_path)
        except FileExistsError:
            pass

        original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        imsave(self.output_path + "/Original Image.jpg", original_image_rgb)

    def grab_cut(self, left, right, top, bottom):
        mask = np.zeros(self.original_image.shape[:2], dtype="uint8")
        rect = (left, top, right - left, bottom - top)

        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")

        (self.mask, bgModel, fgModel) = cv2.grabCut(self.original_image, mask, rect, bgModel, fgModel, iterCount=10,
                                                    mode=cv2.GC_INIT_WITH_RECT)

        # the output mask has four possible output values, marking each pixel
        # in the mask as (1) definite background, (2) definite foreground,
        # (3) probable background, and (4) probable foreground
        values = (
            ("Definite Background", cv2.GC_BGD),
            ("Probable Background", cv2.GC_PR_BGD),
            ("Definite Foreground", cv2.GC_FGD),
            ("Probable Foreground", cv2.GC_PR_FGD),
        )

        # loop over the possible GrabCut mask values
        for (name, value) in values:
            # construct a mask that for the current value
            print("Showing mask for '{}'".format(name))
            valueMask = (mask == value).astype("uint8") * 255

            plt.imshow(valueMask)
            plt.title(name)
            plt.imsave(self.output_path + name + ".jpg", valueMask)
            plt.show()

    def plot_outputs(self, model_type):
        # we'll set all definite background and probable background pixels
        # to 0 while definite foreground and probable foreground pixels are
        # set to 1
        outputMask = np.where((self.mask == cv2.GC_BGD) | (self.mask == cv2.GC_PR_BGD), 0, 1)
        # scale the mask from the range [0, 1] to [0, 255]
        outputMask = (outputMask * 255).astype("uint8")
        # apply a bitwise AND to the image using our mask generated by
        # GrabCut to generate our final output image
        output = cv2.bitwise_and(self.original_image, self.original_image, mask=outputMask)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        plt.imsave(self.output_path + "GrabCut" + str(model_type) + ".jpg", output)

        image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.title("Original Image")
        plt.show()

        outputMask = cv2.cvtColor(outputMask, cv2.COLOR_BGR2RGB)
        plt.imshow(outputMask)
        plt.title("Output Mask")
        plt.show()

        plt.imshow(output)
        plt.title("Output Image")
        plt.show()

class Colouring:
    def __init__(self, name, output, path):
        self.file_name = name
        self.output_path = output
        self.file_path = path
        self.original_image = cv2.imread(self.file_path)

    def calculate_optimal_clusters(self):
        Sum_of_squared_distances = []
        times = []
        # Dimension of the original image
        rows, cols, _ = self.original_image.shape

        # Flatten the image
        imag = np.array(self.original_image).reshape(rows * cols, 3)

        K = range(5, 31, 5)
        for k in K:
            print(k)
            start = perf_counter()
            km = KMeans(n_clusters=k)
            km.fit(imag)
            stop = perf_counter()
            times.append(stop-start)
            Sum_of_squared_distances.append(km.inertia_)

            #  Replace each pixel value with its nearby centroid
            compressed_image = km.cluster_centers_[km.labels_]
            compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

            #  Reshape the image to original dimension
            compressed_image = compressed_image.reshape(rows, cols, 3)
            compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)

            imsave((str(output_path) + str(k) + "Clustered.jpg"), compressed_image)

        kneedle = KneeLocator(K, Sum_of_squared_distances, S=1.0, curve="convex", direction="decreasing")
        plot_elbow_method(Sum_of_squared_distances, times, K, knee=kneedle.knee, output_path=self.output_path)

        # Implement k-means clustering to form k clusters
        k_means = KMeans(n_clusters=kneedle.knee)
        k_means.fit(imag)

        #  Replace each pixel value with its nearby centroid
        compressed_image = k_means.cluster_centers_[k_means.labels_]
        self.compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

        #  Reshape the image to original dimension
        compressed_image = self.compressed_image.reshape(rows, cols, 3)
        self.compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)

        imsave((self.output_path + self.file_name + str(kneedle.knee) + "Clustered.jpg"), self.compressed_image)

    def thresholding(self):
        # The output of the k-means algorithm is an array in the form of Blue,Green,Red (BGR) which needs to be converted to grayscale before we conduct a Gaussian Blur.
        self.compressed_img = cv2.cvtColor(self.compressed_image, cv2.COLOR_BGR2GRAY)

        k = 9  # Size of the kernel for the Blur
        # blurred = cv2.GaussianBlur(self.compressed_img, (k, k), sigmaX=(k - 1) / 6)
        blurred = cv2.bilateralFilter(self.compressed_img, k, 75, 75)

        # Conduct Adaptive Thresholding to the blurred image
        threshold_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 1)
        self.threshold_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, (15, 15))
        imsave((self.output_path + "Threshold Gray.jpg"), threshold_image, cmap="gray")

    def median(self, filter_size=3):
        self.threshold_image = cv2.medianBlur(self.threshold_image, filter_size)
        imsave((self.output_path + "Threshold Gray Bil Filter Median Blur.jpg"), self.threshold_image, cmap="gray")


file_names = [os.path.basename(x) for x in glob.glob('MultiInputs/*.jpg')]

for file_name in file_names:
    file_name = file_name.split(".", 2)[0]
    file_path = str("MultiInputs/" + file_name + ".jpg")
    directory_creator(file_name=file_name)
    output_path = ("MultiOutput/" + file_name + "/")

    FasterRCNN = RegionDetector(file_name=file_name, file_path=file_path, path=output_path)
    FasterRCNN.load_img()
    FasterRCNN.load_model()
    left, right, top, bottom = FasterRCNN.draw_boxes(max_boxes=6)

    cutter = GrabCut(name=file_name, file_path=file_path, path=output_path)
    cutter.create_directory()
    cutter.grab_cut(int(left), int(right), int(top), int(bottom))
    cutter.plot_outputs(model_type="Faster RCNN")
    file_path = "MultiOutput/" + file_name + "/" + "GrabCutBoundingBox" + ".jpg"

    colouring = Colouring(name=file_name, output=output_path, path=file_path)
    colouring.calculate_optimal_clusters()
    colouring.thresholding()
    colouring.median()

