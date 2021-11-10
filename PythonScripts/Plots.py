import matplotlib.pyplot as plt


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
    plt.show()
