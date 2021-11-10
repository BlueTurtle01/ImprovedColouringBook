import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
Takes a random image from the Values.csv file and plots n different configurations of that file and asks the user
to select their favourite output. This will allow us to get a tally of the most popular configurations. Using this tally
we can then train a neural network to estimate the optimal settings for kernel_size, clusters, and sigma so that in future
the user does not need to adjust any of the settings.
"""


class ImageSelect:
    def __init__(self, n):
        self.data = pd.read_csv('Values.csv')
        all_names = self.data['ImageID'].tolist()
        random_id = random.randint(0, len(all_names)-1)
        self.random_file = all_names[random_id]
        self.all_files_by_name = self.data[self.data['ImageID'] == self.random_file]
        self.clusters = self.all_files_by_name['Clusters']
        self.filter_size = self.all_files_by_name['Filter size']
        self.sigma = self.all_files_by_name['Sigma']
        self.unique_colours = self.all_files_by_name['Unique Colours']

        # Now we have all the rows relating to 1 random image we need to choose n of those rows.
        self.random_row_ids = sorted(random.sample(range(0, self.all_files_by_name.shape[0]), n))

    def collate_files(self):
        for id in self.random_row_ids:
            file_name = 'MLOutputs/' + str(self.random_file) + 'Clusters' + str(self.clusters.iloc[id]) + 'FilterSize' \
                        + str(self.filter_size.iloc[id]) + 'Sigma' + str(self.sigma.iloc[id]) + 'UniqueColours' +\
                        str(self.unique_colours.iloc[id]) + ".jpg"
            imag = mpimg.imread(file_name)
            plt.imshow(imag)
            plt.xticks([])
            plt.yticks([])
            plt.title('Image: ' + str(id))
            plt.show()

    def user_choice(self):
        choice = input("Enter the number of the image you prefer:")
        row_number_in_full_df = self.random_row_ids[int(choice)]
        current_score = self.data.iloc[row_number_in_full_df, 5]
        self.data.iloc[row_number_in_full_df, 5] = current_score + 1
        print(self.data)
        self.data.to_csv('Values.csv', index=False)


file_finder = ImageSelect(n=4)
file_finder.collate_files()
file_finder.user_choice()
