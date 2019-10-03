"""Datasets module"""
import os
import logging
import sys
import torch
from torch.utils.data import Dataset
from histocartography.io.wsi import WSI
from histocartography.io.annotations import ImageAnnotation
from histocartography.io.annotations import XMLAnnotation
from histocartography.io.annotations import CSVAnnotation

ANNOTATION_LOADER = {
    '.png': ImageAnnotation,
    '.xml': XMLAnnotation,
    '.csv': CSVAnnotation
}


# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::ml::datasets')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
h1.setFormatter(formatter)
log.addHandler(h1) 


# class WSIPatchSegmentationDataset(IterableDataset):

#     def __init__(
#         self,
#         input_files,
#         label_files,
#         patch_size,
#         stride,
#         input_fn=None,
#         label_fn=None,
#         label_names=None
#     ):
#         """
#             Dataset of WSI for patch-based segmentation.
#             This Iterable Dataset (requires pytorch 1.2.0) yields patches from
#             two paired lists of files:
#             input_files (list): list of files with the input images
#             label_files (list): list of files (paired with input) with the
#                 corresponding labels
#             patch_size (tuple): desired patch size
#             stride (tuple): desired stride to collect the patches

#             Optionally:
#             input_fn (function): function to be applied to every input patch
#             label_fn (function): function to be applied to every label patch
#         """

#         self.pairs = zip(input_files, label_files)
#         self.input_fn = input_fn
#         self.label_fn = label_fn
#         self.current_wsi = None
#         self.patch_size = patch_size
#         self.stride = stride
#         self.label_names = label_names

#     def __iter__(self):
#         for wsi_filename, label_filename in self.pairs:

#             _, file_extension = os.path.splitext(label_filename)
#             loaded_labels = ANNOTATION_LOADER[file_extension](label_filename)
#             wsi_image = WSI(wsi_filename, loaded_labels)
#             for patch_data in wsi_image.patches(
#                 size=self.patch_size, stride=self.stride, annotations=True
#             ):
#                 x, y, full_width, full_height, x_mag, y_mag, patch, labels = patch_data

#                 if self.label_fn is not None:
#                     labels = self.label_fn(labels)

#                 if self.input_fn is not None:
#                     patch = self.input_fn(patch)

#                 yield patch, labels


class WSIPatchSegmentationDataset(Dataset):

    def __init__(
        self,
        input_file,
        label_file,
        patch_size,
        stride,
        mag=1,
        input_fn=None,
        label_fn=None,
        label_names=None
    ):
        """
            Dataset of WSI for patch-based segmentation.
            This Iterable Dataset (requires pytorch 1.2.0) yields patches from
            two paired lists of files:
            input_files (list): list of files with the input images
            label_files (list): list of files (paired with input) with the
                corresponding labels
            patch_size (tuple): desired patch size
            stride (tuple): desired stride to collect the patches

            Optionally:
            input_fn (function): function to be applied to every input patch
            label_fn (function): function to be applied to every label patch
        """

        _, file_extension = os.path.splitext(label_file)
        loaded_labels = ANNOTATION_LOADER[file_extension](label_file)
        
        self.patch_size = patch_size
        self.label_fn = label_fn
        self.input_fn = input_fn

        self.wsi = WSI(input_file, loaded_labels)
        self.patches_info = self.wsi.patch_positions(
            (0, 0), patch_size, stride, mag
        )


    def __getitem__(self, index):

        downsample, level, xy_positions = self.patches_info
        
        patch, labels = self.wsi.get_patch_with_labels(
            downsample, level, xy_positions[index], self.patch_size
        )

        if self.label_fn is not None:
            labels = self.label_fn(labels)

        if self.input_fn is not None:
            patch = self.input_fn(patch)


        # RGBA to RGB and channels first
        return torch.from_numpy(patch[:,:,0:3]).permute(2,0,1).float(), torch.from_numpy(labels).float().unsqueeze(0)

    def __len__(self):

        downsample, level, xy_positions = self.patches_info

        return len(xy_positions)