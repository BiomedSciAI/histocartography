import glob
import cv2
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
from utils import *

class Extract_Patches:
    def __init__(self, config, is_balanced):
        self.base_img_path = config.base_img_path
        self.base_patches_path = config.base_patches_path
        self.tumor_types = config.tumor_types
        self.is_balanced = is_balanced

        self.patch_size_40x = config.patch_size          #patch size at 40x
        self.patch_size_20x = 2 * config.patch_size      #patch size at 20x
        self.patch_size_10x = 4 * config.patch_size      #patch size at 10x
        self.offset_20x = config.patch_size
        self.offset_40x = int(1.5 * config.patch_size)
        self.offset_centroid = int(self.patch_size_10x/ 2)

        if self.is_balanced:
            self.stride = [1.5*config.patch_size, 3*config.patch_size, 1.5*config.patch_size, 1.5*config.patch_size,
                           config.patch_size, 3*config.patch_size, 3*config.patch_size]
        else:
            self.stride = [self.patch_size_10x for i in range(len(self.tumor_types))]

        self.stride = [int(x) for x in self.stride]

        for t in self.tumor_types:
            create_directory(self.base_patches_path + t)
            create_directory(self.base_patches_path + t + '/10x/')
            create_directory(self.base_patches_path + t + '/20x/')
            create_directory(self.base_patches_path + t + '/40x/')

    def extract_patches(self):
        total_patches = 0

        for t in range(len(self.tumor_types)):
            if self.tumor_types[t] != 'dcis':
                continue

            print('Extracting patches for: ', self.tumor_types[t])

            img_paths = glob.glob(self.base_img_path + self.tumor_types[t] + '/*.png')
            img_paths.sort()

            total_patches_per_tumor = 0
            for i in range(len(img_paths)):
                if i%20 == 0:
                    print(i, '/', len(img_paths))

                imgname = os.path.basename(img_paths[i]).split('.')[0]
                img_ = Image.open(img_paths[i])
                img = np.array(img_)
                img_.close()

                (h, w, c) = img.shape
                if h < self.patch_size_10x:
                    img = np.pad(img, ((0, self.patch_size_10x), (0, 0), (0, 0)), mode='constant', constant_values=255)

                if w < self.patch_size_10x:
                    img = np.pad(img, ((0, 0), (0, self.patch_size_10x), (0, 0)), mode='constant', constant_values=255)

                counter = self.select_patches(imgname, img, self.tumor_types[t])
                if counter == 0:
                    print('ZERO patches: ', imgname, img.shape)
                total_patches += counter
                total_patches_per_tumor += counter


            print('#Patches per tumor:', total_patches_per_tumor, '\n')

        print('TOTAL:', total_patches)
        
    def select_patches(self, imgname, img, tumor_type, is_visualize=False):
        (h, w, c) = img.shape
        counter = 0
        if is_visualize:
            fig, ax = plt.subplots(1)
            ax.imshow(img)

        x = 0
        while (x + self.patch_size_10x) <= w:
            y = 0
            while (y + self.patch_size_10x) <= h:
                img_10x = cv2.resize(img[y : y + self.patch_size_10x,
                                         x: x + self.patch_size_10x, :],
                                     (self.patch_size_40x, self.patch_size_40x),
                                     interpolation=cv2.INTER_NEAREST)

                img_20x = cv2.resize(img[y + self.offset_20x : y + self.offset_20x + self.patch_size_20x,
                                         x + self.offset_20x : x + self.offset_20x + self.patch_size_20x, :],
                                     (self.patch_size_40x, self.patch_size_40x),
                                     interpolation=cv2.INTER_NEAREST)

                img_40x = img[y + self.offset_40x : y + self.offset_40x + self.patch_size_40x,
                              x + self.offset_40x : x + self.offset_40x + self.patch_size_40x, :]

                Image.fromarray(img_10x).save(self.base_patches_path + tumor_type + '/10x/' + imgname + '_' + str(counter)
                                              + '_' + str(x + self.offset_centroid) + '_' + str(y + self.offset_centroid) + '.png')

                Image.fromarray(img_20x).save(self.base_patches_path + tumor_type + '/20x/' + imgname + '_' + str(counter)
                                              + '_' + str(x + self.offset_centroid) + '_' + str(y + self.offset_centroid) + '.png')

                Image.fromarray(img_40x).save(self.base_patches_path + tumor_type + '/40x/' + imgname + '_' + str(counter)
                                              + '_' + str(x + self.offset_centroid) + '_' + str(y + self.offset_centroid) + '.png')

                if is_visualize:
                    rect = Rectangle((x, y), self.patch_size_10x, self.patch_size_10x,
                                     linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    rect = Rectangle((x + self.offset_20x, y + self.offset_20x), self.patch_size_20x, self.patch_size_20x,
                                     linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
                    rect = Rectangle((x + self.offset_40x, y + self.offset_40x), self.patch_size_40x, self.patch_size_40x,
                                     linewidth=2, edgecolor='b', facecolor='none')
                    ax.add_patch(rect)

                counter += 1
                y += self.stride[self.tumor_types.index(tumor_type)]

            x += self.stride[self.tumor_types.index(tumor_type)]

        if is_visualize:
            plot(img)

        return counter


