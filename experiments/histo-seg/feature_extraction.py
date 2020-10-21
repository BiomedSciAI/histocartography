from abc import abstractmethod
from typing import Tuple, Union

import numpy as np
import torch
import torchvision
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
from tqdm.auto import tqdm

from utils import PipelineStep


class FeatureExtractor(PipelineStep):
    def __init__(self, **kwargs) -> None:
        """Abstract class that extracts features from superpixels"""
        super().__init__(**kwargs)

    def process(self, input_image: np.ndarray, superpixels: np.ndarray) -> torch.Tensor:
        """Extract features from the input_image for the defined superpixels

        Args:
            input_image (np.array): Original RGB image
            superpixels (np.array): Extracted superpixels

        Returns:
            torch.Tensor: Extracted features
        """
        return self._extract_features(input_image, superpixels)

    @abstractmethod
    def _extract_features(
        self, input_image: np.ndarray, superpixels: np.ndarray
    ) -> torch.Tensor:
        """Extract features from the input_image for the defined superpixels

        Args:
            input_image (np.array): Original RGB image
            superpixels (np.array): Extracted superpixels

        Returns:
            torch.Tensor: Extracted features
        """


class HandcraftedFeatureExtractor(FeatureExtractor):
    """Helper class to extract handcrafted features from superpixels"""

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Extract handcrafted features from images"""
        super().__init__(**kwargs)
        import cv2

    def _extract_features(
        self, input_image: np.ndarray, superpixels: np.ndarray
    ) -> torch.Tensor:
        """Extract handcrafted features from the input_image in the defined superpixel regions

        Args:
            input_image (np.array): Original RGB Image
            superpixels (np.array): Extracted superpixels

        Returns:
            torch.Tensor: Extracted features
        """
        node_feat = []

        img_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        img_square = np.square(input_image)

        # -------------------------------------------------------------------------- Entropy per channel
        img_entropy = Entropy(img_gray, disk(3))

        # For each super-pixel
        regions = regionprops(superpixels)

        for _, region in enumerate(regions):
            sp_mask = np.array(superpixels == region["label"], np.uint8)
            sp_rgb = cv2.bitwise_and(input_image, input_image, mask=sp_mask)
            sp_gray = img_gray * sp_mask
            mask_size = np.sum(sp_mask)
            mask_idx = np.where(sp_mask != 0)

            # -------------------------------------------------------------------------- CONTOUR-BASED SHAPE FEATURES
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

            # -------------------------------------------------------------------------- COLOR FEATURES
            # (rgb color space) [13 x 3 features]
            def color_features_per_channel(img_rgb_ch, img_rgb_sq_ch):
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

            # enddef

            feats_r = color_features_per_channel(sp_rgb[:, :, 0], img_square[:, :, 0])
            feats_g = color_features_per_channel(sp_rgb[:, :, 1], img_square[:, :, 1])
            feats_b = color_features_per_channel(sp_rgb[:, :, 2], img_square[:, :, 2])
            feats_color = [feats_r, feats_g, feats_b]
            feats_color = [item for sublist in feats_color for item in sublist]

            # -------------------------------------------------------------------------- TEXTURE FEATURES
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

            # -------------------------------------------------------------------------- STACKING ALL FEATURES
            sp_feats = feats_shape + feats_color + feats_texture

            features = np.hstack(sp_feats)
            node_feat.append(features)
        # endfor

        node_feat = np.vstack(node_feat)
        return torch.Tensor(node_feat)


class SuperpixelPatchDataset(Dataset):
    """Helper class to use a give image and extracted superpixels as a dataset"""

    def __init__(
        self,
        image: np.ndarray,
        superpixels: np.ndarray,
        size: int,
        fill_value: Union[int, None],
    ) -> None:
        """Create a dataset for a given image and extracted superpixel with desired patches of (size, size, 3).
            If fill_value is not None, it fills up pixels outside the superpixel with this value (all channels)

        Args:
            image (np.ndarray): RGB input image
            superpixels (np.ndarray): Extracted superpixels
            size (int): Desired size of patches
            fill_value (Union[int, None]): Value to fill outside the superpixels (None means do not fill)
        """
        self.image = image
        self.superpixel = superpixels
        self.properties = regionprops(superpixels)
        self.dataset_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.patch_size = (size, size, 3)
        self.fill_value = fill_value

    def _get_superpixel_patch(self, region_property: _RegionProperties) -> np.ndarray:
        """Returns the image patch with the correct padding for a given region property

        Args:
            region_property (_RegionProperties): Region property of the superpixel

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

        # Handle no mask scenario and too large superpixels
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
        mask_region = (self.superpixel != region_property.label)[
            min_x:max_x, min_y:max_y
        ]
        image_region[mask_region] = self.fill_value
        output_image[
            image_top_left[0] : image_top_left[0] + x_length,
            image_top_left[1] : image_top_left[1] + y_length,
        ] = image_region
        return output_image

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor]:
        """Loads an image for a given superpixel index

        Args:
            index (int): Superpixel index

        Returns:
            Tuple[int, torch.Tensor]: superpixel_index, image as tensor
        """
        input_image = self._get_superpixel_patch(self.properties[index])
        transformed_image = self.dataset_transform(Image.fromarray(input_image))
        return self.properties[index].label - 1, transformed_image  # It needs -1 since skimage starts at 1 for some reason

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
            architecture (str): String of architecture. Can be [resnet{18,34,50,101,152}, vgg{16,19}]
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
            architecture (str): Name of architecture. Can be [resnet{18,34,50,101,152}, vgg{16,19}]

        Returns:
            Tuple[nn.Module, int]: The model and the number of features
        """
        if "resnet" in architecture:
            if "18" in architecture:
                model = torchvision.models.resnet18(pretrained=True)
            elif "34" in architecture:
                model = torchvision.models.resnet34(pretrained=True)
            elif "50" in architecture:
                model = torchvision.models.resnet50(pretrained=True)
            elif "101" in architecture:
                model = torchvision.models.resnet101(pretrained=True)
            elif "152" in architecture:
                model = torchvision.models.resnet152(pretrained=True)
            else:
                raise NotImplementedError("ERROR: Select from Resnet: 34, 50, 101, 152")
            num_features = list(model.children())[-1].in_features
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
        elif "vgg" in architecture:
            if "16" in architecture:
                model = torchvision.models.vgg16_bn(pretrained=True)
            elif "19" in architecture:
                model = torchvision.models.vgg19_bn(pretrained=True)
            else:
                raise NotImplementedError("ERROR: Select from VGG: 16, 19")
            classifier = list(model.classifier.children())[:1]
            num_features = list(model.classifier.children())[-1].in_features
            model.classifier = nn.Sequential(*classifier)
        else:
            raise NotImplementedError("ERROR: Select from resnet, vgg")
        return model, num_features

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Computes the embedding of a normalized image input

        Args:
            image (torch.Tensor): Normalized image input

        Returns:
            torch.Tensor: Embedding of image
        """
        with torch.no_grad():
            embeddings = self.model(image).squeeze()
            embeddings = embeddings.cpu().detach()
            return embeddings


class DeepFeatureExtractor(FeatureExtractor):
    """Helper class to extract deep features from superpixels"""

    def __init__(
        self,
        architecture: str,
        mask: bool = True,
        size: int = 224,
        batch_size: int = 32,
        num_workers: int = 0,
        **kwargs,
    ) -> None:
        """Create a deep feature extractor

        Args:
            architecture (str): Name of the architecture to use. Can be [resnet{18,34,50,101,152}, vgg{16,19}]
            mask (bool, optional): Whether to mask out the parts outside the superpixel. Defaults to True.
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
        self, input_image: np.ndarray, superpixels: np.ndarray
    ) -> torch.Tensor:
        """Extract features for a given RGB image and its extracted superpixels

        Args:
            input_image (np.ndarray): RGB input image
            superpixels (np.ndarray): Extracted superpixels

        Returns:
            torch.Tensor: Extracted features
        """
        image_dataset = SuperpixelPatchDataset(
            input_image, superpixels, self.size, self.fill_value
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
        for i, image_batch in image_loader:
            image_batch = image_batch.to(self.device)
            embeddings = self.patch_feature_extractor(image_batch)
            features[i, :] = embeddings
        return features.cpu().detach()
