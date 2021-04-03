import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave


def plot_image(image, output_path, title, _map="gray"):
    plt.imshow(image, cmap=_map)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    imsave((str(output_path) + str(title) + ".jpg"), image, cmap=_map)
    plt.show()


def plot_errors(error_df, sum_of_squared_distances, k):
    plt.plot(k, sum_of_squared_distances, 'bx-')
    plt.plot(k, error_df["Difference"])
    plt.xlabel('k')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def plot_histogram(imag):
    plt.hist(imag.ravel()/255., 256)  # Histogram is not bimodal so we can not use Otsu's thresholding.
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Histogram for input image")
    plt.show()
