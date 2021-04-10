import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave


def plot_image(image, output_path, title, _map="gray"):
    plt.imshow(image, cmap=_map)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    imsave((str(output_path) + str(title) + ".jpg"), image, cmap=_map)
    plt.show()


def plot_elbow_method(Sum_of_squared_distances, times, K, knee):
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.plot(K, Sum_of_squared_distances, color=color)
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Sum of Squared Distances', color=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.plot(K, times, color=color)
    ax2.set_ylabel("Execution Time (Seconds)", color=color)

    plt.title('Elbow Method for optimal k')
    plt.axvline(knee, label="Elbow Point", color="green")
    plt.savefig(str(output_path) + "/ElbowMethod.png")
    plt.show()


def plot_histogram(imag):
    plt.hist(imag.ravel()/255., 256)  # Histogram is not bimodal so we can not use Otsu's thresholding.
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Histogram for input image")
    plt.show()
