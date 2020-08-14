import os
from matplotlib import pyplot as plt


def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
# enddef


def plot(img, cmap=''):
    if cmap == '':
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.show()
# enddef
