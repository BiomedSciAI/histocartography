import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity
from matplotlib.patches import Rectangle


class PatchExtraction:
    def __init__(self, config):
        self.base_img_path = config.base_img_path
        self.base_patches_path = config.base_patches_path
        self.tumor_types = config.tumor_types
        self.base_data_split_path = config.base_data_split_path

        self.patch_size = config.patch_size
        self.patch_threshold = int(0.1 * self.patch_size * self.patch_size)
        self.patch_stride = int(self.patch_size / 1)

        self.mode = ['train', 'val', 'test']

        for m in range(len(self.mode)):
            print('Working on ', self.mode[m], '..........')
            self.create_directory(config.base_patches_path + self.mode[m])

            total_patches = 0
            for t in range(len(self.tumor_types)):
                print(self.tumor_types[t])

                filename = self.base_data_split_path + \
                    self.mode[m] + '_list_' + self.tumor_types[t] + '.txt'
                img_paths = []
                with open(filename, 'r') as f:
                    for line in f:
                        line = line.split('\n')[0]
                        if line != '':
                            path = self.base_img_path + \
                                self.tumor_types[t] + '/' + line + '.png'
                            img_paths.append(path)

                img_paths.sort()

                self.patches_save_path = config.base_patches_path + \
                    self.mode[m] + '/' + config.tumor_types[t] + '/'
                self.create_directory(self.patches_save_path)

                for i in range(len(img_paths)):
                    self.imgname = os.path.basename(img_paths[i]).split('.')[0]
                    img_ = Image.open(img_paths[i])
                    img_rgb = np.array(img_)
                    img_.close()

                    counter = self.select_all_patches(img_rgb)

                    # nuclei_mask = self.get_nuclei_mask(img_rgb)
                    # counter = self.select_patches(img_rgb, nuclei_mask)
                    if counter == 0:
                        print(self.imgname, ':', counter)

                    total_patches += counter

            print('TOTAL:', total_patches)

    def create_directory(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def get_nuclei_mask(self, img):
        img_h = rgb2hed(img)[:, :, 0]
        img_h = rescale_intensity(img_h, out_range=(0, 255)).astype(np.uint8)
        ret, img = cv2.threshold(
            img_h, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img

    def select_patches(self, img_rgb, mask):
        mask = mask / 255.0
        (h, w, c) = img_rgb.shape
        counter = 0
        x = 0
        while (x + self.patch_size) < w:
            y = 0
            while (y + self.patch_size) < h:
                mask_ = mask[y: y + self.patch_size, x: x + self.patch_size]
                if np.sum(mask_) > self.patch_threshold:
                    img_ = img_rgb[y: y + self.patch_size,
                                   x: x + self.patch_size, :]

                    Image.fromarray(img_).save(
                        self.patches_save_path + self.imgname + '_' + str(counter) + '.png')
                    counter += 1

                y += self.patch_stride

            x += self.patch_stride

        return counter

    def select_all_patches(self, img_rgb):
        (h, w, c) = img_rgb.shape
        counter = 0
        x = 0
        while (x + self.patch_size) < w:
            y = 0
            while (y + self.patch_size) < h:
                img_ = img_rgb[y: y + self.patch_size,
                               x: x + self.patch_size, :]

                # debug purposes 
                print('Save new patch at coord x: {} | y: {} | w/h: {}'.format(x, y, self.patch_size))
                print('Save path:', self.patches_save_path + self.imgname + '_' + str(counter) + '.png')

                Image.fromarray(img_).save(
                    self.patches_save_path +
                    self.imgname +
                    '_' +
                    str(counter) +
                    '.png')
                counter += 1
                y += self.patch_stride

            x += self.patch_stride

        return counter

    def plot(self, img, cmap=''):
        if cmap == '':
            plt.imshow(img)
        else:
            plt.imshow(img, cmap=cmap)
        plt.show()

    def show_selected_patches(self, img_rgb, loc):
        fig, ax = plt.subplots(1)
        ax.imshow(img_rgb)
        for i in range(loc.shape[0]):
            rect = Rectangle((loc[i,
                                  1],
                              loc[i,
                                  0]),
                             self.patch_size,
                             self.patch_size,
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)

        plt.show()
