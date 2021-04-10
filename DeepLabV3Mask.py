import os
import tarfile
import tempfile
from six.moves import urllib
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from matplotlib.pyplot import imsave
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

        imsave(self.output_path + "SegmentationMaskDeepLab.jpg", arr=seg_image)


file_name = "david"
file_path = "MultiInputs/" + file_name + ".jpg"
output_path = "MultiOutput/" + file_name + "/"

Mask = MaskFinder(file_path=file_path, file_name=file_name, output_path="MultiOutput/")
Mask.define_model()
Mask.run()
Mask.vis_segmentation()
