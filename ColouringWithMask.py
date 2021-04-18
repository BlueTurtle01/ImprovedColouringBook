from matplotlib.pyplot import imsave
from sklearn.cluster import KMeans
import pandas as pd
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
import tarfile
import tempfile
from six.moves import urllib
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


class MaskFinder:
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, file_path, file_name, output_path):
        self.original_im = Image.open(file_path)
        self.file_name = file_name
        self.output_path = output_path
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        MODEL_NAME = 'mobilenetv2_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

        _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
        _MODEL_URLS = {
          'mobilenetv2_coco_voctrainaug':
            'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
          'mobilenetv2_coco_voctrainval':
            'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
          'xception_coco_voctrainaug':
            'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
          'xception_coco_voctrainval':
            'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
        }

        _TARBALL_NAME = 'deeplab_model.tar.gz'

        model_dir = tempfile.mkdtemp()
        tf.gfile.MakeDirs(model_dir)

        download_path = os.path.join(model_dir, _TARBALL_NAME)
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(download_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def define_model(self):
        self.LABEL_NAMES = np.asarray({
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
        })

        FULL_LABEL_MAP = np.arange(21).reshape(21, 1)
        self.FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    def run(self):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = self.original_im.size
        self.resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        self.target_size = (int(self.resize_ratio * width), int(self.resize_ratio * height))

        resized_image = self.original_im.convert('RGB').resize(self.target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        self.seg_map = batch_seg_map[0]

    def vis_segmentation(self):
        original_width = self.target_size[0] / self.resize_ratio
        original_height = self.target_size[1] / self.resize_ratio

        seg_map_pil = Image.fromarray(np.uint8(self.seg_map))
        seg_map_pil = np.array(seg_map_pil.resize((int(original_width), int(original_height))))
        seg_image = label_to_color_image(seg_map_pil).astype(np.uint8)

        imsave(self.output_path + self.file_name + "/SegmentationMaskDeepLab.jpg", arr=seg_image)


class GrabCutMask:
    def __init__(self, name, file_path, path):
        self.file_name = name
        self.output_path = path
        self.file_path = file_path
        self.original_image = cv2.imread(self.file_path)
        self.mask = cv2.imread("MultiOutput/" + str(file_name) + "/SegmentationMaskDeepLab.jpg", cv2.IMREAD_GRAYSCALE)
        self.mask_height, self.mask_width = self.mask.shape[0], self.mask.shape[1]

    def crop_center(self):
        cropx = self.mask_width
        cropy = self.mask_height
        y, x, _ = self.original_image.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        self.original_image = self.original_image[starty:starty + cropy, startx:startx + cropx, :]

    def grab_cut_mask(self):
        self.mask[self.mask > 0] = cv2.GC_PR_FGD
        self.mask[self.mask == 0] = cv2.GC_BGD
        plt.imshow(self.mask)
        plt.show()

        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")

        (self.mask, bgModel, fgModel) = cv2.grabCut(self.original_image, self.mask, None, bgModel, fgModel, iterCount=10, mode=cv2.GC_INIT_WITH_MASK)

        values = (
            ("Definite Background", cv2.GC_BGD),
            ("Probable Background", cv2.GC_PR_BGD),
            ("Definite Foreground", cv2.GC_FGD),
            ("Probable Foreground", cv2.GC_PR_FGD),
        )


        for (name, value) in values:
            # construct a mask that for the current value
            print("Showing mask for '{}'".format(name))
            valueMask = (self.mask == value).astype("uint8") * 255

            plt.imshow(valueMask)
            plt.title(name)
            plt.imsave(self.output_path + name + "GrabCutMask.jpg", valueMask)
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

    # Create Mask
    Mask = MaskFinder(file_path=file_path, file_name=file_name, output_path="MultiOutput/")
    Mask.define_model()
    Mask.run()
    Mask.vis_segmentation()

    #GrabCut with mask
    mask_cutter = GrabCutMask(name=file_name, file_path=file_path, path=output_path)
    mask_cutter.crop_center()
    mask_cutter.grab_cut_mask()
    mask_cutter.plot_outputs(model_type="Mask")

    file_path = "MultiOutput/" + file_name + "/" + "GrabCutMask.jpg"
    colouring = Colouring(name=file_name, output=output_path, path=file_path)
    colouring.calculate_optimal_clusters()
    colouring.thresholding()
    colouring.median()

