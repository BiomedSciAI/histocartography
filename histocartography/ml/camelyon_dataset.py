import os
import sys
import logging

import numpy as np

from torch.utils.data import Dataset
from PIL import Image

from histocartography.io.wsi import WSI
from histocartography.io.annotations import ASAPAnnotation


log = logging.getLogger('Histocartography::ml::CamelyonDataset')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.DEBUG)
log.addHandler(h1)

class CamelyonDataset(Dataset):

    def __init__(self,
                 wsi_path,
                 data_path,
                 num_patches,
                 patch_size,
                 patch_mag):
        """
        Dataset class to handle Camelyon16 WSIs.
        On first call creates num_patches for each wsi and
        class(for tumor slide). It does so for all WSIs at wsi_path and stores
        the results in data_path/normal and data_path/tumor.

        On future calls it provides a way to dynamically load the patches used
        for training in pytorch.
        """
        self.data_path = data_path
        self.wsi_path = wsi_path
        self.num_patches = num_patches
        
        self.patch_size = patch_size
        self.patch_mag = patch_mag

        self.tumor_patches_path = os.path.join(data_path, 'tumor')
        self.normal_patches_path = os.path.join(data_path, 'normal')

        # try to load the data
        self.load_data()

        # if it doesn't exist, create it from wsi_path images
        self.prepare_data()

    def prepare_data(self):
        """Creates the patches from the camelyon WSIs."""

        self._check_create_dirs([self.data_path,
                                 self.tumor_patches_path,
                                 self.normal_patches_path])

        # iterate over all WSI's
        wsi_train_path = os.path.join(self.wsi_path, 'training')
        wsi_train_tumor = os.listdir(os.path.join(wsi_train_path,
                                                  'tumor'))
        wsi_train_normal = os.listdir(os.path.join(wsi_train_path,
                                                   'normal'))
        self._iterate_through_wsis(wsi_train_tumor,
                                   wsi_train_path,
                                   True)
        self._iterate_through_wsis(wsi_train_normal,
                                   wsi_train_path,
                                   False)

    def _iterate_through_wsis(self,
                              wsi_list,
                              wsi_path,
                              tumor=False):

        for idx, wsi in enumerate(wsi_list):
            if (wsi.endswith('.tif')):
                wsi_name,_ = os.path.splitext(wsi)
                log.debug(wsi_name)
                annotation = None
                if(tumor):
                    annotation = ASAPAnnotation(
                        os.path.join(wsi_path,
                                     'lesion_annotations',
                                     f'{wsi_name}.xml'))
                    img = WSI(os.path.join(wsi_path, 'tumor', wsi),
                            annotations=annotation, minimum_tissue_content=0.1)
                else:
                    img = WSI(os.path.join(wsi_path, 'normal', wsi),
                              minimum_tissue_content=0.4)
                self._generate_patches(img, wsi_name, tumor)

    def _generate_patches(self, img, wsi_name, tumor=False):
        size = (self.patch_size, self.patch_size)
        if tumor:
            data_path = self.tumor_patches_path
        else:
            data_path = self.normal_patches_path

        patches = img.patches(mag=self.patch_mag,
                              size=size,
                              stride=size,
                              use_label_mask=tumor)

        self._save_patches(patches, wsi_name, data_path) 
        # if tumor, call this function again to generate non tumor patches from
        # the same wsi
        if tumor:
            self._generate_patches(img, wsi_name, tumor=False)

    def _save_patches(self, patches, wsi_name, data_path):
        i = 0
        for patch in patches:
            i += 1
            im = Image.fromarray(patch['region'])
            filename = '{}_{}_{}_{}.png'.format(wsi_name,
                                                patch['mag'],
                                                patch['x'],
                                                patch['y'])
            im.save(os.path.join(data_path, filename))
            #im = Image.fromarray(patch['annotation']*255)
            #im.save(os.path.join(data_path, f'label_{filename}'))
            if i==self.num_patches:
                return


    def _check_create_dirs(self, dirs):
        for path in dirs:
            if not os.path.exists(path):
                os.makedirs(path)


    def load_data(self):
        pass
