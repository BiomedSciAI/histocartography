"""Detect and Classify nuclei from an image with the HoverNet model."""

from pathlib import Path
from typing import Tuple
from mlflow.pytorch import load_model
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2 

from skimage.measure import regionprops
from skimage.morphology import remove_small_objects, watershed
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes

from ..pipeline import PipelineStep
from ..utils.image import extract_patches_from_image
from ..utils.io import is_mlflow_url


class NucleiExtractor(PipelineStep):
    """Nuclei extraction"""

    def __init__(
        self,
        model_path: str,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        """Create a nuclei extractor

        Args:
            model_path (str): Path to the pre-trained model or mlflow URL
            batch_size (int, optional): Batch size. Defaults to 32.
        """
        super().__init__(**kwargs)

        # set class attributes 
        self.model_path = model_path
        self.batch_size = batch_size

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract nuclei from the input_image

        Args:
            input_image (np.array): Original RGB image

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: instance_map, instance_centroids  
        """
        return self._extract_nuclei(input_image)

    def _extract_nuclei(
        self, input_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from the input_image for the defined structure

        Args:
            input_image (np.array): Original RGB image

        Returns:
            Tuple[np.ndarray, np.ndarray]: instance_map, instance_centroids 
        """

        image_dataset = ImageToPatchDataset(input_image)

        def collate(batch):
            coords = [x[0] for x in batch]
            patches = torch.stack([x[1] for x in batch])
            return coords, patches

        image_loader = DataLoader(
            image_dataset,
            shuffle=False,
            batch_size=self.batch_size,
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

        # post process instance map 
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

    def precompute(self, final_path) -> None:
        """Precompute all necessary information"""
        if self.base_path is not None:
            self._link_to_path(Path(final_path) / "nuclei_maps")


class ImageToPatchDataset(Dataset):
    """Helper class to transform an image as a set of patched wrapped in a pytorch dataset"""

    def __init__(
        self,
        image: np.ndarray,
    ) -> None:
        """Create a dataset for a given image and extracted instance maps with desired patches.
           Patches have shape of (3, 256, 256) as defined by HoverNet model.

        Args:
            image (np.ndarray): RGB input image
        """
        self.image = image
        self.dataset_transform = transforms.Compose(
            [
                transforms.ToTensor(),
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


def process_np_hv_channels(pred):
    """
    Process Nuclei Prediction with XY Coordinate Map

    Args:
        pred (np.ndarray): HoverNet model output, that contains: 
                            - channel 0 contain probability map of nuclei
                            - channel 1 contains X-map
                            - channel 2 contains Y-map
    Returns:
         pred_instance (np.ndarray): instance map 
    """

    blb_raw = pred[:, :, 0]  # extract proba maps
    h_dir_raw = pred[:, :, 1]  # extract horizontal map
    v_dir_raw = pred[:, :, 2]  # extract vertical map

    # Processing @TODO: use one-liner
    blb = np.copy(blb_raw)
    blb[blb >= 0.5] = 1
    blb[blb <  0.5] = 0

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1 # background is 0 already

    # @TODO: over-write variable, ie get rid of clear variables 
    h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    h_dir_raw = None  # clear variable
    v_dir_raw = None  # clear variable

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)
    h_dir = None  # clear variable
    v_dir = None  # clear variable

    sobelh = 1 - (cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    sobelv = 1 - (cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

    overall = np.maximum(sobelh, sobelv)
    sobelh = None  # clear variable
    sobelv = None  # clear variable
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    # nuclei values form peaks so inverse to get basins
    dist = -cv2.GaussianBlur(dist,(3, 3),0)

    overall[overall >= 0.5] = 1
    overall[overall <  0.5] = 0
    marker = blb - overall
    overall = None # clear variable
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)
 
    pred_inst = watershed(dist, marker, mask=blb, watershed_line=False)
    return pred_inst


def process_instance(pred_map, output_dtype='uint16'):
    """
    Post processing script for image tiles

    Args:
        pred_map (np.ndarray): commbined output of np and hv branches
        output_dtype (str): data type of output
    
    Returns:
        pred_inst (np.ndarray): pixel-wise nuclear instance segmentation prediction
    """

    pred_inst = np.squeeze(pred_map)
    pred_inst = process_np_hv_channels(pred_inst)
    pred_inst = pred_inst.astype(output_dtype)
    return pred_inst
