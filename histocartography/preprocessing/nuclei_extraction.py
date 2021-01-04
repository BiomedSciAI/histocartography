"""Detect and Classify nuclei from an image with the HoverNet model."""

from abc import abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from mlflow.pytorch import load_model
from skimage.measure import regionprops
from tqdm import tqdm

from ..utils import dynamic_import_from
from .pipeline import PipelineStep
from ..utils.image import extract_patches_from_image
from ..utils.hover import process_instance
from ..utils.io import is_mlflow_url
# from ..ml.models.hovernet import HoVerNet


class NucleiExtractor(PipelineStep):
    """Nuclei extraction"""

    def __init__(
        self,
        model_path: str,
        size: int = 256,
        batch_size: int = 2,
        num_workers: int = 0,
        **kwargs,
    ) -> None:
        """Create a nuclei extractor

        Args:
            model_path (str): Path to the pre-trained model or mlflow URL
            size (int, optional): Desired size of patches. Defaults to 224.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, 0): Number of workers. Defaults to 0. 
        """
        super().__init__(**kwargs)

        # set class attributes 
        self.model_path = model_path
        self.size = size
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.num_workers in [0, 1]:
            torch.set_num_threads(1)

        # handle GPU
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")

        # load nuclei extraction model 
        if is_mlflow_url(model_path):
            self.model = load_model(model_path,  map_location=torch.device('cpu'))
        else:
            self.model = torch.load(model_path)

        self.model.eval()
        self.model = self.model.to(self.device)

    def process(
        self,
        input_image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract nuclei from the input_image

        Args:
            input_image (np.array): Original RGB image

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: instance_map, , instance_labels, instance_centroids  
        """
        return self._extract_nuclei(input_image)

    def _extract_nuclei(
        self, input_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract features from the input_image for the defined structure

        Args:
            input_image (np.array): Original RGB image

        Returns:
            Tuple[np.ndarray, np.ndarray]: instance_map, instance_labels, instance_centroids 
        """

        image_dataset = ImageToPatchDataset(input_image, self.size)

        def collate(batch):
            coords = [x[0] for x in batch]
            patches = torch.stack([x[1] for x in batch])
            return coords, patches

        image_loader = DataLoader(
            image_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate
        )
        pred_map = torch.empty(
            size=(image_dataset.max_x_coord, image_dataset.max_y_coord, 3),
            dtype=torch.float32,
            device=self.device,
        )

        for coords, image_batch in tqdm(image_loader, desc='Patch-level nuclei detection'):
            image_batch = image_batch.to(self.device)
            with torch.no_grad():
                out = self.model(image_batch).cpu()
                for i in range(out.shape[0]):
                    left = coords[i][0] # left, bottom, right, top
                    bottom = coords[i][1]
                    right = coords[i][2]
                    top = coords[i][3]
                    pred_map[bottom:top, left:right, :] = out[i, :, :, :]

        # crop to original image size
        pred_map = pred_map.cpu().detach().numpy()
        pred_map = pred_map[:image_dataset.im_h,:image_dataset.im_w, :]

        # post process instance map and labels 
        instance_map = process_instance(pred_map)

        # extract the centroid location in the instance map
        regions = regionprops(instance_map)
        instance_centroids = np.empty((len(regions), 2))
        for i, region in enumerate(regions):
            center_x, center_y = region.centroid
            center_x = int(round(center_x))
            center_y = int(round(center_y))
            instance_centroids[i, 0] = center_x
            instance_centroids[i, 1] = center_y
        
        return instance_map, instance_centroids


class ImageToPatchDataset(Dataset):
    """Helper class to transform an image as a set of patched wrapped in a pytorch dataset"""

    def __init__(
        self,
        image: np.ndarray,
        size: int,
    ) -> None:
        """Create a dataset for a given image and extracted instance maps with desired patches.
           Patches have shape of (3, size, size).

        Args:
            image (np.ndarray): RGB input image
            size (int): Desired size of patches
        """
        self.image = image
        self.dataset_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.im_h = image.shape[0] 
        self.im_w = image.shape[1]
        self.all_patches, self.coords = extract_patches_from_image(image, self.im_h, self.im_w)
        self.nr_patches = len(self.all_patches)
        self.max_y_coord = max([coord[-2] for coord in self.coords])  
        self.max_x_coord = max([coord[-1] for coord in self.coords])

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor]:
        """Loads an image for a given instance maps index

        Args:
            index (int): patch index

        Returns:
            Tuple[int, torch.Tensor]: index, image as tensor
        """
        patch = self.all_patches[index]
        coord = self.coords[index]
        transformed_image = self.dataset_transform(Image.fromarray(patch))
        return coord, transformed_image

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return self.nr_patches
