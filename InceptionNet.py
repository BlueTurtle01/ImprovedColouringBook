# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub

# For downloading the image.
import matplotlib.pyplot as plt

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
import cv2


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


class RegionDetector():
    def __init__(self, file_name, file_path, path):
        self.file_name = file_name
        self.file_path = file_path
        self.output_path = path
        self.model = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
        self.detector = hub.load(self.model).signatures['default']

    def load_img(self):
        #img = tf.io.read_file(self.file_path)
        self.img = cv2.imread(self.file_path)
        self.width, self.height, _ = self.img.shape
        #self.img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

        #self.img = tf.image.decode_jpeg(img, channels=3)

    def load_model(self):
        converted_img = tf.image.convert_image_dtype(self.img, tf.float32)[tf.newaxis, ...]
        result = self.detector(converted_img)
        self.result = {key: value.numpy() for key, value in result.items()}
        print("Found %d objects." % len(self.result["detection_scores"]))

    def draw_boxes(self, max_boxes=1, min_score=0.5):
        global ymin, xmin, ymax, xmax, image_
        boxes = self.result["detection_boxes"]
        class_names = self.result["detection_class_entities"]
        scores = self.result["detection_scores"]

        """Overlay labeled boxes on an image with formatted scores and label names."""
        colors = list(ImageColor.colormap.values())
        font = ImageFont.load_default()

        for i in range(min(boxes.shape[0], max_boxes)):
            if scores[i] >= min_score:
                ymin, xmin, ymax, xmax = tuple(boxes[i])
                display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                               int(100 * scores[i]))
                color = colors[hash(class_names[i]) % len(colors)]
                image_ = Image.fromarray(np.uint8(self.img)).convert("RGB")
                draw_bounding_box_on_image(image_, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
                np.copyto(self.img, np.array(image_))

        image_ = cv2.cvtColor(np.array(image_), cv2.COLOR_BGR2RGB)
        plt.imshow(image_)
        plt.show()

        return ymin*self.height, xmin*self.width, ymax*self.height, xmax*self.width

rnn = RegionDetector(file_name="horse.jpg", file_path="MultiInputs/horse.jpg", path="MultiOutput/horse/")
rnn.load_img()
rnn.load_model()
ymin, xmin, ymax, xmax = rnn.draw_boxes(max_boxes=1)

print(round(ymin), 0)

