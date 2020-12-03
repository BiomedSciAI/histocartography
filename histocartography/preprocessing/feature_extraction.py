"""Extract features from images for a given structure"""

from abc import abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from histocartography.utils import dynamic_import_from
from PIL import Image
from scipy.stats import skew
from skimage.feature import greycomatrix, greycoprops
from skimage.filters.rank import entropy as Entropy
from skimage.measure import regionprops
from skimage.measure._regionprops import _RegionProperties
from skimage.morphology import disk
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .pipeline import PipelineStep


class FeatureExtractor(PipelineStep):
    """Base class for feature extraction"""

    def process(
        self, input_image: np.ndarray, instance_map: np.ndarray
    ) -> torch.Tensor:
        """Extract features from the input_image for the defined instance_map

        Args:
            input_image (np.array): Original RGB image
            instance_map (np.array): Extracted instance_map

        Returns:
            torch.Tensor: Extracted features
        """
        return self._extract_features(input_image, instance_map)

    @abstractmethod
    def _extract_features(
        self, input_image: np.ndarray, instance_map: np.ndarray
    ) -> torch.Tensor:
        """Extract features from the input_image for the defined structure

        Args:
            input_image (np.array): Original RGB image
            structure (np.array): Structure to extract features

        Returns:
            torch.Tensor: Extracted features
        """


class HandcraftedFeatureExtractor(FeatureExtractor):
    """Helper class to extract handcrafted features from instance maps"""

    @staticmethod
    def _color_features_per_channel(img_rgb_ch, img_rgb_sq_ch, mask_idx, mask_size):
        codes = img_rgb_ch[mask_idx[0], mask_idx[1]].ravel()
        hist, _ = np.histogram(codes, bins=np.arange(0, 257, 32))  # 8 bins
        feats_ = list(hist / mask_size)
        color_mean = np.mean(codes)
        color_std = np.std(codes)
        color_median = np.median(codes)
        color_skewness = skew(codes)

        codes = img_rgb_sq_ch[mask_idx[0], mask_idx[1]].ravel()
        color_energy = np.mean(codes)

        feats_.append(color_mean)
        feats_.append(color_std)
        feats_.append(color_median)
        feats_.append(color_skewness)
        feats_.append(color_energy)
        return feats_

    def _extract_features(
        self, input_image: np.ndarray, instance_map: np.ndarray
    ) -> torch.Tensor:
        """Extract handcrafted features from the input_image in the defined instance_map regions

        Args:
            input_image (np.array): Original RGB Image
            instance_map (np.array): Extracted instance_map. Different regions have different int values,
                                     the background is defined to have value 0 and is ignored.

        Returns:
            torch.Tensor: Extracted shape, color and texture features:
                          Shape:   area, convex_area, eccentricity, equivalent_diameter, euler_number, extent, filled_area,
                                   major_axis_length, minor_axis_length, orientation, perimiter, solidity;
                          Color:   Per channel (RGB) histogram with 8 bins:
                                   mean, std, median, skewness, energy;
                          Texture: entropy, glcm_contrast, glcm_dissililarity, glcm_homogeneity, glcm_energy, glcm_ASM
                                   (glcm = grey-level co-occurance matrix);
        """
        node_feat = []

        img_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        img_square = np.square(input_image)

        img_entropy = Entropy(img_gray, disk(3))

        # For each super-pixel
        regions = regionprops(instance_map)

        for _, region in enumerate(regions):
            sp_mask = np.array(instance_map == region["label"], np.uint8)
            sp_rgb = cv2.bitwise_and(input_image, input_image, mask=sp_mask)
            sp_gray = img_gray * sp_mask
            mask_size = np.sum(sp_mask)
            mask_idx = np.where(sp_mask != 0)

            # Compute using mask [12 features]
            area = region["area"]
            convex_area = region["convex_area"]
            eccentricity = region["eccentricity"]
            equivalent_diameter = region["equivalent_diameter"]
            euler_number = region["euler_number"]
            extent = region["extent"]
            filled_area = region["filled_area"]
            major_axis_length = region["major_axis_length"]
            minor_axis_length = region["minor_axis_length"]
            orientation = region["orientation"]
            perimeter = region["perimeter"]
            solidity = region["solidity"]
            feats_shape = [
                area,
                convex_area,
                eccentricity,
                equivalent_diameter,
                euler_number,
                extent,
                filled_area,
                major_axis_length,
                minor_axis_length,
                orientation,
                perimeter,
                solidity,
            ]

            # (rgb color space) [13 x 3 features]
            feats_r = self._color_features_per_channel(
                sp_rgb[:, :, 0], img_square[:, :, 0], mask_idx, mask_size
            )
            feats_g = self._color_features_per_channel(
                sp_rgb[:, :, 1], img_square[:, :, 1], mask_idx, mask_size
            )
            feats_b = self._color_features_per_channel(
                sp_rgb[:, :, 2], img_square[:, :, 2], mask_idx, mask_size
            )
            feats_color = [feats_r, feats_g, feats_b]
            feats_color = [item for sublist in feats_color for item in sublist]

            # Entropy (gray color space) [1 feature]
            entropy = cv2.mean(img_entropy, mask=sp_mask)[0]

            # GLCM texture features (gray color space) [5 features]
            glcm = greycomatrix(sp_gray, [1], [0])
            # Filter out the first row and column
            filt_glcm = glcm[1:, 1:, :, :]

            glcm_contrast = greycoprops(filt_glcm, prop="contrast")
            glcm_contrast = glcm_contrast[0, 0]
            glcm_dissimilarity = greycoprops(filt_glcm, prop="dissimilarity")
            glcm_dissimilarity = glcm_dissimilarity[0, 0]
            glcm_homogeneity = greycoprops(filt_glcm, prop="homogeneity")
            glcm_homogeneity = glcm_homogeneity[0, 0]
            glcm_energy = greycoprops(filt_glcm, prop="energy")
            glcm_energy = glcm_energy[0, 0]
            glcm_ASM = greycoprops(filt_glcm, prop="ASM")
            glcm_ASM = glcm_ASM[0, 0]

            feats_texture = [
                entropy,
                glcm_contrast,
                glcm_dissimilarity,
                glcm_homogeneity,
                glcm_energy,
                glcm_ASM,
            ]

            sp_feats = feats_shape + feats_color + feats_texture

            features = np.hstack(sp_feats)
            node_feat.append(features)

        node_feat = np.vstack(node_feat)
        return torch.Tensor(node_feat)


class InstanceMapPatchDataset(Dataset):
    """Helper class to use a give image and extracted instance maps as a dataset"""

    def __init__(
        self,
        image: np.ndarray,
        instance_map: np.ndarray,
        size: int,
        fill_value: Optional[int],
    ) -> None:
        """Create a dataset for a given image and extracted instance maps with desired patches
           of (size, size, 3). If fill_value is not None, it fills up pixels outside the
           instance maps with this value (all channels)

        Args:
            image (np.ndarray): RGB input image
            instance maps (np.ndarray): Extracted instance maps
            size (int): Desired size of patches
            fill_value (Optional[None]): Value to fill outside the instance maps
                                         (None means do not fill)
        """
        self.image = image
        self.instance_map = instance_map
        self.properties = regionprops(instance_map)
        self.dataset_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.patch_size = (size, size, 3)
        self.fill_value = fill_value

    def _get_instance_patch(self, region_property: _RegionProperties) -> np.ndarray:
        """Returns the image patch with the correct padding for a given region property

        Args:
            region_property (_RegionProperties): Region property of the instance maps

        Returns:
            np.ndarray: Representative image patch
        """
        # Prepare input and output data
        output_image = np.ones(self.patch_size, dtype=np.uint8)
        if self.fill_value is not None:
            output_image *= self.fill_value
        else:
            output_image *= 255  # Have a white background in case we are at the border

        # Extract center
        center_x, center_y = region_property.centroid
        center_x = int(round(center_x))
        center_y = int(round(center_y))

        # Extract only super pixel
        if self.fill_value is not None:
            min_x, min_y, max_x, max_y = region_property.bbox
            x_length = max_x - min_x
            y_length = max_y - min_y

        # Handle no mask scenario and too large instance maps
        if self.fill_value is None or x_length > self.patch_size[0]:
            min_x = center_x - (self.patch_size[0] // 2)
            max_x = center_x + (self.patch_size[0] // 2)

        if self.fill_value is None or y_length > self.patch_size[1]:
            min_y = center_y - (self.patch_size[1] // 2)
            max_y = center_y + (self.patch_size[1] // 2)

        # Handle border cases
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(self.image.shape[0], max_x)
        max_y = min(self.image.shape[1], max_y)
        x_length = max_x - min_x
        y_length = max_y - min_y
        assert x_length <= self.patch_size[0]
        assert y_length <= self.patch_size[1]

        # Actual image copying
        image_top_left = (
            ((self.patch_size[0] - x_length) // 2),
            ((self.patch_size[1] - y_length) // 2),
        )
        image_region = self.image[min_x:max_x, min_y:max_y]
        mask_region = (self.instance_map != region_property.label)[
            min_x:max_x, min_y:max_y
        ]
        if self.fill_value is not None:
            image_region[mask_region] = self.fill_value
        output_image[
            image_top_left[0] : image_top_left[0] + x_length,
            image_top_left[1] : image_top_left[1] + y_length,
        ] = image_region
        return output_image

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor]:
        """Loads an image for a given instance maps index

        Args:
            index (int): Instance index

        Returns:
            Tuple[int, torch.Tensor]: instance_index, image as tensor
        """
        input_image = self._get_instance_patch(self.properties[index])
        transformed_image = self.dataset_transform(Image.fromarray(input_image))
        return index, transformed_image

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.properties)


class PatchFeatureExtractor:
    """Helper class to use a CNN to extract features from an image"""

    def __init__(self, architecture: str) -> None:
        """Create a patch feature extracter of a given architecture and put it on GPU if available

        Args:
            architecture (str): String of architecture. According to torchvision.models syntax
        """
        self.model, self.num_features = self._select_model(architecture)
        self.model.eval()

        # Handle GPU
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")
        self.model = self.model.to(self.device)

    @staticmethod
    def _select_model(architecture: str) -> Tuple[nn.Module, int]:
        """Returns the model and number of features for a given name

        Args:
            architecture (str): Name of architecture. According to torchvision.models syntax

        Returns:
            Tuple[nn.Module, int]: The model and the number of features
        """
        model_class = dynamic_import_from("torchvision.models", architecture)
        model = model_class()
        if isinstance(model, torchvision.models.resnet.ResNet):
            feature_dim = model.fc.in_features
            model.fc = nn.Sequential()
        else:
            feature_dim = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential()
        return model, feature_dim

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        """Computes the embedding of a normalized image input

        Args:
            image (torch.Tensor): Normalized image input

        Returns:
            torch.Tensor: Embedding of image
        """
        with torch.no_grad():
            embeddings = self.model(patch).squeeze()
            embeddings = embeddings.cpu().detach()
            return embeddings


class DeepFeatureExtractor(FeatureExtractor):
    """Helper class to extract deep features from instance maps"""

    def __init__(
        self,
        architecture: str,
        mask: bool = False,
        size: int = 224,
        batch_size: int = 32,
        num_workers: int = 0,
        **kwargs,
    ) -> None:
        """Create a deep feature extractor

        Args:
            architecture (str): Name of the architecture to use. According to torchvision.models syntax
            mask (bool, optional): Whether to mask out the parts outside the instance maps. Defaults to True.
            size (int, optional): Desired size of patches. Defaults to 224.
        """
        self.architecture = architecture
        self.mask = mask
        self.size = size
        super().__init__(**kwargs)
        self.patch_feature_extractor = PatchFeatureExtractor(self.architecture)
        self.fill_value = 255 if self.mask else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.num_workers in [0, 1]:
            torch.set_num_threads(1)

        # Handle GPU
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")

    def _extract_features(
        self, input_image: np.ndarray, instance_map: np.ndarray
    ) -> torch.Tensor:
        """Extract features for a given RGB image and its extracted instance_map

        Args:
            input_image (np.ndarray): RGB input image
            instance_map (np.ndarray): Extracted instance_map

        Returns:
            torch.Tensor: Extracted features
        """
        image_dataset = InstanceMapPatchDataset(
            input_image, instance_map, self.size, self.fill_value
        )
        image_loader = DataLoader(
            image_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        features = torch.empty(
            size=(len(image_dataset), self.patch_feature_extractor.num_features),
            dtype=torch.float32,
            device=self.device,
        )
        for i, image_batch in tqdm(image_loader, desc='Instance-level feature extraction'):
            image_batch = image_batch.to(self.device)
            embeddings = self.patch_feature_extractor(image_batch)
            features[i, :] = embeddings
        return features.cpu().detach()
