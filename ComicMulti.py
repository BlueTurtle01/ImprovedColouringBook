from matplotlib.pyplot import imread, imshow, imsave
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import pandas as pd
import os
import glob


def plot_histogram(imag):
    plt.hist(imag.ravel()/255., 256)  # Histogram is not bimodal so we can not use Otsu's thresholding.
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Histogram for input image")
    plt.show()


def plot_image(image, title, map="gray"):
    plt.imshow(image, cmap=map)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    imsave((str(output_path) + str(title) + ".jpg"), image, cmap=map)
    plt.show()


def plot_errors(error_df, Sum_of_squared_distances, K):
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.plot(K, error_df["Difference"])
    plt.xlabel('k')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def white_to_transparency():
    img = (str(output_path) + "Threshold Gray.jpg")
    img = Image.open(img)

    x = np.asarray(img.convert('RGBA')).copy()

    # Find all cells where any of the pixels in RGB space are not white - i.e have some colour
    x[:, :, 3] = (255 * (x[:, :, :3] < 220).any(axis=2)).astype(np.uint8)

    return Image.fromarray(x)


def calculate_font_size(caption, box, img_fraction, font_type="comicbold.ttf"):
    fontsize = 8  # Starting fontsize
    font = ImageFont.truetype(font_type, fontsize)
    while font.getsize(caption)[0] < img_fraction * box.size[0]:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype(font_type, fontsize)

    fontsize -= 1
    font = ImageFont.truetype(font_type, fontsize)

    return font


def add_caption(comic_frame):
    width, height = comic_frame.size

    draw = ImageDraw.Draw(comic_frame)
    caption = "Not all superheroes are found in comic books"
    tag = "#HeroesOfThePandemic"

    bottom_right_caption = (width, height)
    top_left_caption = (width*0.25, height*0.9)
    top_left_date = (0, 0)
    bottom_right_date = (width*.40, height*0.1)

    # x0, y1 start at the top left of the image
    # x0, y0 is the top left of the caption box
    # x1, y1 is the bottom right of the caption box

    # Create a caption box at the top left of the image
    draw.rectangle((top_left_date, bottom_right_date), fill="white", outline="black", width=6)
    tag_font_size = calculate_font_size(tag, comic_frame, img_fraction=0.3, font_type="bada.ttf")
    draw.text((width*0.05, height*0.03), tag, (0, 0, 0), font=tag_font_size)

    # Create a white caption box at the bottom right of the image
    draw.rectangle((top_left_caption, bottom_right_caption), fill="white", outline="black", width=6)
    caption_font_size = calculate_font_size(caption, comic_frame, img_fraction=0.60, font_type="anime.ttf")
    draw.text((top_left_caption[0]*1.2, height*.94), caption, (0, 0, 0), font=caption_font_size)
    comic_frame.save(str(output_path) + "Caption.png")



def blend_outputs(clusters):
    """
    Some of the edges are not closed loops so it can be confusing where one colour stops and another starts.
    I think overlaying the outlines onto a slightly transparent version of the compressed image gives a good guide to the user for which colour to fill each area with.

    :param clusters: Number of clusters from the k-means algorithm
    :return: NA
    """

    compressed_image = imread((str(output_path) + str(file_name) + str(clusters) + "Clustered.jpg"))
    compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)
    compressed_rgba = cv2.cvtColor(compressed_image, cv2.COLOR_RGB2RGBA)
    compressed_PIL = Image.fromarray(compressed_rgba)

    # Change the outline to RGBA and then convert to PIL Image
    outline_rgba = white_to_transparency()

    # Combine the outline image and the output of the k-means alogrithm together.
    alpha_blended = Image.alpha_composite(compressed_PIL, outline_rgba)
    alpha_blended.save((str(output_path) + "Comic Character.png"))

    # Add a caption box over the composite image to give it a comic book effect.
    add_caption(alpha_blended)


    plt.show()



def main_func(img, clusters):
    """
    :param img:
    :param clusters: Number of clusters
    :return: NA
    """

    # The output of the k-means algorithm is an array in the form of Blue,Green,Red (BGR) which needs to be converted to grayscale before we conduct a Gaussian Blur.
    compressed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # imread reads images as Blue, Green, Red, Alpha
    plot_image(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), title="Compressed")

    k = 9  # Size of the kernal for the Gaussian Blur
    blurred = cv2.GaussianBlur(compressed_img, (k, k), sigmaX=(k-1)/6)
    plot_image(image=blurred, title="Blurred", map="gray")

    # Conduct Adaptive Thresholding to the blurred image
    threshold_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 3)
    threshold_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, (15, 15))

    # Overlay the threshold_image on top of a slightly transparent copy of the clustered image.
    plot_image(threshold_image, title="Threshold Gray")
    blend_outputs(clusters=clusters)

    return threshold_image


def calculate_clusters(file_path):
    # Credit: https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
    original_image = cv2.imread(file_path)

    # Create a directory for the outputs if one does not already exist
    try:
        os.mkdir("MultiOutput/" + str(file_name))
    except FileExistsError:
        pass

    imsave("MultiOutput/" + str(file_name) + "/Original Image.jpg", original_image)

    # Save the original image in RGB
    plot_image(image=cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), title="Original Image")
    plot_histogram(original_image)

    Sum_of_squared_distances = []
    # Dimension of the original image
    rows, cols, _ = original_image.shape

    # Flatten the image
    imag = np.array(original_image).reshape(rows*cols, 3)

    K = range(20, 21)
    for k in K:
        km = KMeans(n_clusters=k)
        km.fit(imag)
        Sum_of_squared_distances.append(km.inertia_)

    error_df = pd.DataFrame(list(zip(K, Sum_of_squared_distances)), columns=["K", "Error"])
    error_df["Difference"] = error_df["Error"] - error_df["Error"].shift(-1)

    plot_errors(error_df, Sum_of_squared_distances, K)

    try:
        clusters = min(error_df.index[error_df["Difference"] < 3e+06].tolist())
    except ValueError:
        clusters = max(error_df["K"])

    # Implement k-means clustering to form k clusters
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(imag)

    #  Replace each pixel value with its nearby centroid
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

    #  Reshape the image to original dimension
    compressed_image = compressed_image.reshape(rows, cols, 3)

    imsave((str(output_path) + str(file_name) + str(clusters) + "Clustered.jpg"), compressed_image)

    return clusters, compressed_image, original_image


file_names = [os.path.basename(x) for x in glob.glob('MultiInputs/*.jpg')]

for file_name in file_names:
    file_name = file_name.split(".", 2)[0]
    file_path = str("MultiInputs/" + file_name + ".jpg")
    output_path = ("MultiOutput/" + str(file_name) + "/")
    clusters, compressed_image, original_image = calculate_clusters(file_path)
    threshold_image = main_func(compressed_image, clusters=clusters)


