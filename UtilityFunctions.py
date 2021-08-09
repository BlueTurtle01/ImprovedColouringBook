import os

def directory_creator(file_name):
    try:
        os.mkdir("MultiOutput/" + str(file_name))
    except FileExistsError:
        pass

