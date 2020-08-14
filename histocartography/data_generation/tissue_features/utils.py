import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from skimage.segmentation import mark_boundaries

def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
#enddef

def plot(img, cmap=''):
    if cmap == '':
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.show()
#enddef

def overlaid_plot(img, sp_map, centroid=None):
    overlaid = np.round(mark_boundaries(img, sp_map, (0, 0, 0)) * 255, 0).astype(np.uint8)

    '''
    if centroid.any() != None:
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        ax.imshow(img)

        for i in range(centroid.shape[0]):
            y = centroid[i, 0]
            x = centroid[i, 1]
            circ = Circle((x, y), 20)
            ax.add_patch(circ)
    #'''

    plt.imshow(overlaid)
    plt.show()
#enddef