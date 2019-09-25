"""Datasets module"""
import os
import torch as nn
from torch.utils.data import IterableDataset
from histocartography.io.wsi import WSI
from histocartography.io.annotations import ImageAnnotation
from histocartography.io.annotations import XMLAnnotation
from histocartography.io.annotations import CSVAnnotation


ANNOTATION_LOADER = {
    '.png': ImageAnnotation,
    '.xml': XMLAnnotation,
    '.csv': CSVAnnotation
}


class WSIPatchSegmentationDataset(IterableDataset):

    def __init__(
        self,
        input_files,
        label_files,
        patch_size,
        stride,
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

        self.pairs = zip(input_files, label_files)
        self.input_fn = input_fn
        self.label_fn = label_fn
        self.current_wsi = None
        self.patch_size = patch_size
        self.stride = stride
        self.label_names = label_names

    def __iter__(self):
        for wsi_filename, label_filename in self.pairs:

            _, file_extension = os.path.splitext(label_filename)
            loaded_labels = ANNOTATION_LOADER[file_extension](label_filename)
            wsi_image = WSI(wsi_filename, loaded_labels)
            for patch_data in wsi_image.patches(
                size=self.patch_size, stride=self.stride, annotations=True
            ):
                x, y, full_width, full_height, x_mag, y_mag, patch, labels = patch_data

                if self.label_fn is not None:
                    labels = self.label_fn(labels)

                if self.input_fn is not None:
                    patch = self.input_fn(patch)
                
                yield patch, labels
