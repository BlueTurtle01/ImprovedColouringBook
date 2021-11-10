import os
from random import choice


def directory_creator(file_name):
    try:
        os.mkdir("MultiOutput/" + str(file_name))
    except FileExistsError:
        pass


def random_values():
    cluster_values = [5, 7, 10, 15, 20]
    sigma_values = [0.33, 0.5]

    clusters, sigma = (choice(cluster_values), choice(sigma_values))

    return clusters, sigma
