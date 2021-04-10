from matplotlib.pyplot import imread, imshow, imsave
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import pandas as pd
from time import perf_counter
from kneed import KneeLocator
from Plots import plot_elbow_method
from UtilityFunctions import directory_creator



file_name = "race"
file_path = str("InputImages/" + file_name + ".jpg")
output_path = ("OutputImages/" + str(file_name) + "/")


def find_colour_pal(compressed_image, im_name):

    """
    This function draws a grid of the colours used for the clusters. This gives the user an easy way to check they have the correct colouring crayons.

    :param compressed_image: Numpy array of the pixel values for the compressed image after applying k-means to the original imput image.
    :param im_name: Name of the file to be used as the filename when saving.
    :return: A palette of the cluster colours
    """
    # Credit: https://stackoverflow.com/a/51729498/4367851
    compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)
    temp_image = Image.fromarray(np.uint8(compressed_image)).convert('RGB')
    colours = Image.Image.getcolors(temp_image)
    df = pd.DataFrame()
    for i in range(len(colours)):
        count, pixel = colours[i]
        red, green, blue = pixel
        hex = '#%02x%02x%02x' % (int(red), int(green), int(blue))
        df = df.append({"Count": int(count), "RGB": pixel, "HEX": hex}, ignore_index=True)

    RGB = df["RGB"].tolist()
    palette = np.array(RGB)[np.newaxis, :, :]

    if palette.shape[0] % 2 == 0:
        palette = palette.reshape(2, -1, 3)
    elif palette.shape[0] % 3 == 0:
        palette = palette.reshape(3, -1, 3)

    plt.figure(figsize=(20, 10))
    plt.imshow(palette)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(("OutputImages/" + str(im_name) + "/Palette.png"), dpi=600, bbox_inches='tight')
    #plt.show()

    return palette


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



def auto_canny(image, sigma=0.5):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def white_to_transparency():
    img = (str(output_path) + "Threshold Gray.jpg")
    img = Image.open(img)

    x = np.asarray(img.convert('RGBA')).copy()

    # Find all cells where any of the pixels in RGB space are not white - i.e have some colour
    x[:, :, 3] = (255 * (x[:, :, :3] < 220).any(axis=2)).astype(np.uint8)

    return Image.fromarray(x)


def calculate_font_size(caption, box, img_fraction):
    fontsize = 20  # Starting fontsize
    font = ImageFont.truetype("comicbold.ttf", fontsize)
    while font.getsize(caption)[0] < img_fraction * box.size[0]:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype("comicbold.ttf", fontsize)

    return font



def add_caption(comic_frame):
    width, height = comic_frame.size
    font = ImageFont.truetype("comicbold.ttf", 40)
    draw = ImageDraw.Draw(comic_frame)
    caption = "Not all superheroes are found in comics"
    tag = "#HeroesOfThePandemic"

    bottom_right_caption = (width, height)
    top_left_caption = (width*0.25, height*0.8)
    top_left_date = (0, 0)
    bottom_right_date = (width*.40, height*0.1)

    # x0, y1 start at the top left of the image
    # x0, y0 is the top left of the caption box
    # x1, y1 is the bottom right of the caption box

    # Create a caption box at the top left of the image
    draw.rectangle((top_left_date, bottom_right_date), fill="white", outline="black", width=6)
    tag_font_size = calculate_font_size(tag, comic_frame, img_fraction=0.2)
    draw.text((20, 20), tag, (0, 0, 0), font=tag_font_size)

    # Create a white caption box at the bottom right of the image
    draw.rectangle((top_left_caption, bottom_right_caption), fill="white", outline="black", width=6)
    caption_font_size = calculate_font_size(caption, comic_frame, img_fraction=0.60)
    #draw.text((top_left_caption[0]+20, top_left_caption[1]+20), "Not all superheroes are found in comics.", (0, 0, 0), font=font)
    #draw.text((top_left_caption[0]+20, top_left_caption[1]+70), "#HeroesOfThePandemic", (0, 0, 0), font=font)
    draw.text((top_left_caption[0]*1.1, (height-top_left_caption[1])*0.3+top_left_caption[1]), caption, (0, 0, 0), font=caption_font_size)
    #draw.text((top_left_caption[0]+20, top_left_caption[1]+70), "#HeroesOfThePandemic", (0, 0, 0), font=font)
    comic_frame.save(str(output_path) + "Caption.png")
    imshow(comic_frame)
    plt.show()


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


def create_painting_template(outline, clusters):
    alpha = 0.2  # This is the amount of the coloured image we include. 0 is none.
    beta = 1 - alpha
    compressed = imread((str(output_path) + str(file_name) + str(clusters) + "Clustered.jpg"))
    compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)

    outline = np.concatenate([outline[..., np.newaxis]] * 3, axis=2)

    output = cv2.addWeighted(outline, beta, compressed, alpha, 0)

    plot_image(output, "Painting Underlay")


def draw_contours(edged):
    """
    Draw the contours over a blank array. The function cv2.DrawContours overlays the contours on top of the bitwise array.
    Which is not ideal if the bitwise array contains some small, noisy contours. Therefore, I created an empty array first and then used this as the base
    for drawing the contours onto.
    :param edged: Output of the Canny Edge Detection algorithm after applying erode and dilation.
    :return:
    """
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    edged = cv2.bitwise_not(edged)
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

    plot_image(contours_, title="Contours")


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
    create_painting_template(threshold_image, clusters=clusters)
    blend_outputs(clusters=clusters)

    edged = auto_canny(blurred)
    edged = cv2.dilate(edged, (20, 20), iterations=1)
    edged = cv2.erode(edged, (20, 20), iterations=1)

    draw_contours(edged)

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

    return threshold_image


def calculate_clusters():
    # Credit: https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
    original_image = cv2.imread(file_path)

    # Create a directory for the outputs if one does not already exist
    directory_creator(file_name=file_name)

    imsave("OutputImages/" + str(file_name) + "/Original Image.jpg", original_image)

    # Save the original image in RGB
    plot_image(image=cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), title="Original Image")
    #plot_histogram(original_image)

    Sum_of_squared_distances = []
    times = []
    # Dimension of the original image
    rows, cols, _ = original_image.shape

    # Flatten the image
    imag = np.array(original_image).reshape(rows*cols, 3)

    K = range(1, 31, 5)
    for k in K:
        start = perf_counter()
        print(k)
        km = KMeans(n_clusters=k)
        km.fit(imag)
        stop = perf_counter()
        times.append((stop-start))
        Sum_of_squared_distances.append(km.inertia_)

        #  Replace each pixel value with its nearby centroid
        compressed_image = km.cluster_centers_[km.labels_]
        compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

        #  Reshape the image to original dimension
        compressed_image = compressed_image.reshape(rows, cols, 3)
        compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB)

        imsave((str(output_path) + str(k) + "Clustered.jpg"), compressed_image)

    kneedle = KneeLocator(K, Sum_of_squared_distances, S=1.0, curve="convex", direction="decreasing")

    # plot_elbow_method: arguments (x, y1, y2, elbow point)
    plot_elbow_method(Sum_of_squared_distances, times, K, knee=kneedle.knee)

    # Implement k-means clustering to form k clusters
    kmeans = KMeans(n_clusters=kneedle.knee)
    kmeans.fit(imag)

    #  Replace each pixel value with its nearby centroid
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

    #  Reshape the image to original dimension
    compressed_image = compressed_image.reshape(rows, cols, 3)
    find_colour_pal(compressed_image, im_name=file_name)

    return kneedle.knee, compressed_image, original_image


def get_image(path, width):
    from reportlab.platypus import Image
    from reportlab.lib import utils
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    aspect = ih / float(iw)
    return Image(path, width=width, height=(width * aspect))


def pdf_creator(original_image):
    from reportlab.platypus import Image as imag
    from reportlab.platypus import PageBreak, NextPageTemplate, BaseDocTemplate, Frame, PageTemplate
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import cm

    elements = []
    # Create a canvas to instantiate the PDF file
    canvas = BaseDocTemplate((str(output_path) + 'ColouringBook.pdf'), pagesize=A4, rightMargin=25, leftMargin=25,
                             topMargin=25, bottomMargin=25)

    portrait_frame = Frame(canvas.leftMargin, canvas.bottomMargin, canvas.width, canvas.height, id='portrait_frame')
    landscape_frame = Frame(canvas.leftMargin, canvas.bottomMargin, canvas.height, canvas.width, id='landscape_frame')

    width = 21
    # Insert the original image
    elements.append(NextPageTemplate('landscape'))
    elements.append(get_image(str("InputImages/" + file_name + ".jpg"), width=width*cm))
    elements.append(PageBreak())

    # Insert the Compressed/clustered image
    elements.append(get_image(("OutputImages/" + str(file_name) + "/Compressed.jpg"), width=width*cm))
    elements.append(PageBreak())

    # Insert the palette of colours required to replicate the clustered output
    #elements.append(imag((str(output_path) + "Palette.png"), width=400, height=50))
    # Insert the outline image
    elements.append(get_image((str(output_path) + "Threshold Gray.jpg"), width=width*cm))
    elements.append(PageBreak())

    # Insert the palette of colours required to replicate the clustered output
    #elements.append(imag((str(output_path) + "Palette.png"), width=400, height=50))
    # Insert the blended image
    elements.append(get_image((str(output_path) + "/Blended.jpg"), width=width*cm))

    canvas.addPageTemplates([PageTemplate(id='portrait', frames=portrait_frame),
                             PageTemplate(id='landscape', frames=landscape_frame, pagesize=landscape(A4))])

    canvas.build(elements)


clusters, compressed_image, original_image = calculate_clusters()
#threshold_image = main_func(compressed_image, clusters=clusters)
#pdf_creator(original_image)

