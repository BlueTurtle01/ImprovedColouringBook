from matplotlib.pyplot import imread, imsave
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import pandas as pd
import os
import glob
from UtilityFunctions import directory_creator

"""
Whilst creating the colouring book image I realised that applying the thresholded outlines on top of the original
clustered image had quite a cartoon strip like aesthetic. I therefore created the below code to blend the two images
together and add a caption and description to the image in the style of a comic book.

It is not included in the final report as it was beyond the aim of the project and would have required more pages.
However, I have included it here for posterity.
"""


def white_to_transparency():
    img = (str(output_path) + "Threshold Gray.jpg")
    img = Image.open(img)

    x = np.asarray(img.convert('RGBA')).copy()

    # Find all cells where any of the pixels in RGB space are not white - i.e have some colour
    x[:, :, 3] = (255 * (x[:, :, :3] < 220).any(axis=2)).astype(np.uint8)

    return Image.fromarray(x)


def calculate_font_size(caption, box, img_fraction, font_type="Fonts/comicbold.ttf"):
    fontsize = 20  # Starting fontsize
    font = ImageFont.truetype(font_type, fontsize)
    while font.getsize(caption)[0] < img_fraction * box.size[0]:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype(font_type, fontsize)

    fontsize -= 1
    font = ImageFont.truetype(font_type, fontsize)

    return font


class ComicCharacter:
    def __init__(self, name, output, path):
        self.file_name = name
        self.output_path = output
        self.file_path = path
        self.original_image = cv2.imread(self.file_path)

    def calculate_optimal_clusters(self):
        Sum_of_squared_distances = []
        # Dimension of the original image
        rows, cols, _ = self.original_image.shape

        # Flatten the image
        imag = np.array(self.original_image).reshape(rows * cols, 3)

        K = range(24, 25)
        for k in K:
            km = KMeans(n_clusters=k)
            km.fit(imag)
            Sum_of_squared_distances.append(km.inertia_)

        error_df = pd.DataFrame(list(zip(K, Sum_of_squared_distances)), columns=["K", "Error"])
        error_df["Difference"] = error_df["Error"] - error_df["Error"].shift(-1)

        try:
            self.clusters = min(error_df.index[error_df["Difference"] < 3e+06].tolist())
        except ValueError:
            self.clusters = max(error_df["K"])

        # Implement k-means clustering to form k clusters
        k_means = KMeans(n_clusters=self.clusters)
        k_means.fit(imag)

        #  Replace each pixel value with its nearby centroid
        self.compressed_image = k_means.cluster_centers_[k_means.labels_]
        self.compressed_image = np.clip(self.compressed_image.astype('uint8'), 0, 255)

        #  Reshape the image to original dimension
        self.compressed_image = self.compressed_image.reshape(rows, cols, 3)

        imsave((self.output_path + self.file_name + str(self.clusters) + "Clustered.jpg"), self.compressed_image)

    def thresholding(self):
        # The output of the k-means algorithm is an array in the form of Blue,Green,Red (BGR) which needs to be converted to grayscale before we conduct a Gaussian Blur.
        self.compressed_img = cv2.cvtColor(self.compressed_image, cv2.COLOR_BGR2GRAY)  # imread reads images as Blue, Green, Red, Alpha

        k = 9  # Size of the kernel for the Gaussian Blur
        #blurred = cv2.GaussianBlur(self.compressed_img, (k, k), sigmaX=(k - 1) / 6)
        blurred = cv2.bilateralFilter(self.compressed_img, k, 75, 75)

        # Conduct Adaptive Thresholding to the blurred image
        threshold_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 3)
        threshold_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, (15, 15))
        imsave((self.output_path + "Threshold Gray.jpg"), threshold_image, cmap="gray")

    def blend_outputs(self):
        """
        Some of the edges are not closed loops so it can be confusing where one colour stops and another starts.
        I think overlaying the outlines onto a slightly transparent version of the compressed image gives a good guide to the user for which colour to fill each area with.
        """

        compressed_image = imread((self.output_path + self.file_name + str(self.clusters) + "Clustered.jpg"))
        compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)
        compressed_rgba = cv2.cvtColor(compressed_image, cv2.COLOR_RGB2RGBA)
        compressed_PIL = Image.fromarray(compressed_rgba)

        # Change the outline to RGBA and then convert to PIL Image
        outline_rgba = white_to_transparency()

        # Combine the outline image and the output of the k-means alogrithm together.
        self.alpha_blended = Image.alpha_composite(compressed_PIL, outline_rgba)
        self.alpha_blended.save((self.output_path + "Comic Character.png"))
        alpha_blended = self.alpha_blended.convert("RGB")
        alpha_blended.save((self.output_path + "Comic Character.pdf"))


        threshold_image = cv2.imread(self.output_path + "/Threshold Gray.jpg")
        imsave((self.output_path + "Threshold Gray.pdf"), threshold_image, cmap="gray")

    def add_caption(self):
        width, height = self.alpha_blended.size

        draw = ImageDraw.Draw(self.alpha_blended)
        caption = "London"
        tag = "April 3rd 2021"

        bottom_right_caption = (width, height)
        top_left_caption = (width * 0.25, height * 0.9)
        top_left_date = (0, 0)
        bottom_right_date = (width * .40, height * 0.1)

        # Create a caption box at the top left of the image
        draw.rectangle((top_left_date, bottom_right_date), fill="white", outline="black", width=6)
        tag_font_size = calculate_font_size(tag, self.alpha_blended, img_fraction=0.3, font_type="Fonts/bada.ttf")
        draw.text((width * 0.05, height * 0.03), tag, (0, 0, 0), font=tag_font_size)

        # Create a white caption box at the bottom right of the image
        draw.rectangle((top_left_caption, bottom_right_caption), fill="white", outline="black", width=6)
        caption_font_size = calculate_font_size(caption, self.alpha_blended, img_fraction=0.60)
        draw.text((top_left_caption[0] * 1.2, height * .93), caption, (0, 0, 0), font=caption_font_size)
        self.alpha_blended.save(self.output_path + "Caption.png")


file_names = [os.path.basename(x) for x in glob.glob('MultiInputs/*.jpg')]

for file_name in file_names:
    file_name = file_name.split(".", 2)[0]
    file_path = str("MultiInputs/" + file_name + ".jpg")
    output_path = ("MultiOutput/" + file_name + "/")

    directory_creator()

    comic = ComicCharacter(name=file_name, output=output_path, path=file_path)
    comic.calculate_optimal_clusters()
    comic.thresholding()
    comic.blend_outputs()
    comic.add_caption()
