import os
import matplotli.pyplot as plt


def directory_creator(file_name):
    try:
        os.mkdir("MultiOutput/" + str(file_name))
    except FileExistsError:
        pass


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


def create_painting_template(outline, clusters):
    alpha = 0.2  # This is the amount of the coloured image we include. 0 is none.
    beta = 1 - alpha
    compressed = imread((str(output_path) + str(clusters) + "Clustered.jpg"))
    compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)

    outline = np.concatenate([outline[..., np.newaxis]] * 3, axis=2)

    output = cv2.addWeighted(outline, beta, compressed, alpha, 0)

    plot_image(output, "Painting Underlay")

def plot_creator(clusters):
    """
    :param n: number of clusters for k-means
    :return: NA - plots
    """
    img, palette = kmeans(int(clusters))

    fig, axs = plt.subplots(3, 2, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.3, wspace=.001)
    axs = axs.ravel()

    for i in range(0, 6):
        if i % 6 == 0:
            img = imread(("OutputImages/" + str(file_name) + str(clusters) + "Clustered.jpg"))
            axs[0].imshow(img.astype(np.uint8))
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            axs[0].set_title("Original with " + str(clusters) + " clusters")

        else:
            axs[i].imshow(canny(img, clusters))
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_title(str("Canny Kernel: " + str(i)))

    plt.savefig("Output" + str(clusters) + ".jpg", dpi=600, bbox_inches='tight')
    plt.show()
