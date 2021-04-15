"""This module handles everything related to stain normalization"""

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from sklearn.decomposition import DictionaryLearning

from ..pipeline import PipelineStep
from .utils import load_image


class StainNormalizer(PipelineStep):
    """Base class for creating fancy stain normalizers"""

    def __init__(
        self,
        target_path: Optional[str] = None,
        precomputed_normalizer_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Create a stain normalizer

        Args:
            target_path (str, optional): Path of the target image for identification
            precomputed_normalizer_path (str, optional): Path of the precomputed normalizer
        """
        assert not(target_path is not None and precomputed_normalizer_path is not None), "Wrong input, provided both targeted and precomputed normalization."

        self.target_path = target_path
        self.precomputed_normalizer_path = precomputed_normalizer_path
        super().__init__(**kwargs)

    @staticmethod
    def _standardize_brightness(input_image: np.ndarray) -> np.ndarray:
        """Standardize an image by moving all values above the 90 percentile to 255

        Args:
            input_image (np.array): Image to normalize

        Returns:
            np.array: Normalized image
        """
        upper_percentile_bound = np.percentile(input_image, 90)
        return np.clip(
            input_image *
            255.0 /
            upper_percentile_bound,
            0,
            255).astype(
            np.uint8)

    @staticmethod
    def _rgb_to_od(input_image: np.ndarray) -> np.ndarray:
        """Convert a given image to the optical density space,
            also sets all values of 0 to 1 to avoid problems in the logarithm

        Args:
            input_image (np.array): Image to convert

        Returns:
            np.array: Input image in the OD space
        """
        mask = input_image == 0
        if mask.any():
            input_image[mask] = 1
        return -1 * np.log(input_image / 255)

    @staticmethod
    def _normalize_rows(input_array: np.ndarray) -> np.ndarray:
        """Normalizes the rows of a given image

        Args:
            input_array (np.array): Image to normalize

        Returns:
            np.array: Normalized image
        """
        return input_array / np.linalg.norm(input_array, axis=1)[:, None]

    def _get_concentrations(
        self, input_image: np.ndarray, stain_matrix: np.ndarray
    ) -> np.ndarray:
        """Extracts the stain concentrations for all pixels of the given input image

        Args:
            input_image (np.array): Input image
            stain_matrix (np.array): Compute stain vectors in the OD color space

        Returns:
            np.array: Extracted stains of all pixels (vectorized image)
        """
        optical_density = (
            self._rgb_to_od(input_image).reshape((-1, 3)).astype(np.float32)
        )
        stain_matrix = stain_matrix.astype(np.float32)
        concentration = np.linalg.lstsq(
            stain_matrix.T, optical_density.T, rcond=-1)[0].T
        return concentration * (concentration > 0)

    @abstractmethod
    def fit(self, target_image: np.ndarray):
        """Fit a normalizer by precomputing the required information

        Args:
            target_image (np.array): Input image
        """

    @abstractmethod
    def _load_values(self, file: h5py.File) -> None:
        """Load the relevant entries from the file

        Args:
            file (h5py.File): Opened file
        """

    @abstractmethod
    def _save_values(self, file: h5py.File) -> None:
        """Saves the relevant entries to a file

        Args:
            file (h5py.File): Opened file
        """

    def _load_precomputed(self) -> None:
        """Loads the precomputed values of a previous fit"""
        if self.normalizer_save_path is not None:
            with h5py.File(self.normalizer_save_path, "r") as input_file:
                self._load_values(input_file)

    def _save_precomputed(self):
        """Saves the precomputed values"""
        if self.normalizer_save_path is not None:
            with h5py.File(self.normalizer_save_path, "w") as output_file:
                self._save_values(output_file)

    @abstractmethod
    def _normalize_image(self, input_image: np.ndarray) -> np.ndarray:
        """Perform the normalization of an image with precomputed values

        Args:
            input_image (np.array): Image to normalize

        Returns:
            np.array: Normalized image
        """

    # type: ignore[override]
    def _process(self, input_image: np.ndarray) -> np.ndarray:
        """Stain normalizes a given image

        Args:
            input_image (np.array): Input image

        Returns:
            np.array: The stain normalized image
        """
        standardized_image = self._standardize_brightness(input_image)
        normalized_image = self._normalize_image(standardized_image)
        return normalized_image

    # type: ignore[override]
    def process_and_save(
            self,
            *args,
            output_name: str,
            **kwargs) -> np.ndarray:
        """Process and save in the provided path as a png image

        Args:
            output_name (str): Name of output file
        """
        assert (
            self.save_path is not None
        ), "Can only save intermediate output if base_path was not None during construction"
        output_path = self.output_dir / f"{output_name}.png"
        if output_path.exists():
            logging.info(
                "%s: Output of %s already exists, using it instead of recomputing",
                self.__class__.__name__,
                output_name,
            )
            try:
                with Image.open(output_path) as input_file:
                    output = np.array(input_file)
            except OSError as error:
                logging.critical("Could not open %s", output_path)
                raise error
        else:
            output = self._process(*args, **kwargs)
            with Image.fromarray(output) as output_image:
                output_image.save(output_path)
        return output

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information

        Args:
            link_path (Union[None, str, Path], optional): Path to link to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save precomputation outputs. Defaults to None.
        """

        self.normalizer_save_path = None

        if self.save_path is not None and link_path is not None:
            self._link_to_path(Path(link_path) / "normalized_images")

        if self.target_path is not None and self.precomputed_normalizer_path is None:
            if self.save_path is not None:
                self.normalizer_save_path = self.output_dir / "normalizer.h5"
            if self.normalizer_save_path is not None:
                if not self.normalizer_save_path.exists():
                    assert (
                        self.target_path is not None
                    ), "Cannot load image if target_path is None"
                    target_image = load_image(Path(self.target_path))
                    self.fit(target_image)
                self._load_precomputed()
            else:
                assert (
                    self.target_path is not None
                ), "Cannot load image if target_path is None"
                target_image = load_image(Path(self.target_path))
                self.fit(target_image)
        elif self.target_path is None and self.precomputed_normalizer_path is not None:
            self.normalizer_save_path = Path(self.precomputed_normalizer_path)
            if (
                self.normalizer_save_path is None
                or not self.normalizer_save_path.exists()
            ):
                raise FileNotFoundError(
                    "Precomputed normalizer does not exist.")
            self._load_precomputed()
        else:
            self.max_concentration_target = []
            self.stain_matrix_target = np.array(
                [[0.5626, 0.7201, 0.4062], [0.2159, 0.8012, 0.5581]]
            ).astype(np.float32)
            self.max_concentration_target = np.array([1.9705, 1.0308]).astype(
                np.float32
            )


class MacenkoStainNormalizer(StainNormalizer):
    """
    Stain normalization based on the method of:
    M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’,
    in 2009 IEEE International Symposium on Biomedical Imaging:
    From Nano to Macro, 2009, pp. 1107–1110.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.15,
        **kwargs,
    ) -> None:
        """Apply the stain normalization with a given target and parameters

        Args:
            alpha (float, optional): Alpha parameter. Defaults to 1.0.
            beta (float, optional): Beta parameter. Defaults to 0.15.
        """
        self.alpha = alpha
        self.beta = beta
        super().__init__(**kwargs)
        # Hidden fields
        self.stain_matrix_key = "stain_matrix"
        self.max_concentration_target_key = "max_concentration_target"

    def _load_values(self, file: h5py.File) -> None:
        """Loads the values computed when fitted

        Args:
            file (h5py.File): Opened file to load from
        """
        try:
            self.stain_matrix_target = file[self.stain_matrix_key][()]
            self.max_concentration_target = file[self.max_concentration_target_key][(
            )]
        except ValueError:
            print(
                f"Invalid stain matrix keys. Expected keys are "
                f"{self.stain_matrix_key} and "
                f"{self.max_concentration_target_key}."
            )

    def _save_values(self, file: h5py.File) -> None:
        """Save values needed to reproduce fit

        Args:
            file (h5py.File): Opened file to save to
        """
        file.create_dataset(
            self.stain_matrix_key,
            data=self.stain_matrix_target,
            compression="gzip",
            compression_opts=9,
        )
        file.create_dataset(
            self.max_concentration_target_key,
            data=self.max_concentration_target,
            compression="gzip",
            compression_opts=9,
        )

    def _get_stain_matrix(self, input_image: np.ndarray) -> np.ndarray:
        """Compute the 2x3 stain matrix with the method from the paper

        Args:
            input_image (np.array): Image to extract the stains from

        Returns:
            np.array: Extracted stains
        """
        optical_density = self._rgb_to_od(input_image).reshape((-1, 3))
        optical_density = optical_density[(
            optical_density > self.beta).any(axis=1), :]
        _, eigvecs = np.linalg.eigh(np.cov(optical_density, rowvar=False))
        eigvecs = eigvecs[:, [2, 1]]
        if eigvecs[0, 0] < 0:
            eigvecs[:, 0] *= -1
        if eigvecs[0, 1] < 0:
            eigvecs[:, 1] *= -1
        that = np.dot(optical_density, eigvecs)
        phi = np.arctan2(that[:, 1], that[:, 0])
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100 - self.alpha)
        v1 = np.dot(eigvecs, np.array([np.cos(min_phi), np.sin(min_phi)]))
        v2 = np.dot(eigvecs, np.array([np.cos(max_phi), np.sin(max_phi)]))
        if v1[0] > v2[0]:
            stain_matrix = np.array([v1, v2])
        else:
            stain_matrix = np.array([v2, v1])
        return self._normalize_rows(stain_matrix)

    def _normalize_image(self, input_image: np.ndarray) -> np.ndarray:
        """Compute the normalization according to the paper

        Args:
            input_image (np.array): Image to normalize

        Returns:
            np.array: Normalized image
        """
        stain_matrix_source = self._get_stain_matrix(input_image)
        source_concentrations = self._get_concentrations(
            input_image, stain_matrix_source
        )
        max_concentration_source = np.percentile(
            source_concentrations, 99, axis=0
        ).reshape((1, 2))
        max_concentration_target = self.max_concentration_target
        source_concentrations *= max_concentration_target / max_concentration_source
        return (255 * np.exp(-1 * np.dot(source_concentrations,
                                         self.stain_matrix_target).reshape(input_image.shape))).astype(np.uint8)

    def fit(self, target_image: np.ndarray) -> None:
        """Fit the normalizer to a target value and save it for the future

        Args:
            target_image (np.array): Target image
        """
        target_image = self._standardize_brightness(target_image)
        self.stain_matrix_target = self._get_stain_matrix(target_image)
        target_concentrations = self._get_concentrations(
            target_image, self.stain_matrix_target
        )
        self.max_concentration_target = np.percentile(
            target_concentrations, 99, axis=0
        ).reshape((1, 2))
        self._save_precomputed()


class VahadaneStainNormalizer(StainNormalizer):
    """
    Stain normalization inspired by method of:
    A. Vahadane et al., ‘Structure-Preserving Color Normalization and Sparse Stain Separation
    for Histological Images’, IEEE Transactions on Medical Imaging,
    vol. 35, no. 8, pp. 1962–1971, Aug. 2016.
    """

    def __init__(
            self,
            threshold: float = 0.8,
            lambda_s: float = 0.1,
            **kwargs) -> None:
        """Create a Vahadame normalizer for a given target image

        Args:
            threshold (float, optional): Threshold for the non-white mask in lab color space.
                                         Defaults to 0.8.
            lambda_s (float, optional): Optimization parameter for the stain extraction.
                                        Defaults to 0.1.
        """
        self.thres = threshold
        self.lambda_s = lambda_s
        super().__init__(**kwargs)
        # Hidden fields
        self.stain_matrix_key = "stain_matrix"

    def _load_values(self, file: h5py.File) -> None:
        """Loads the values computed when fitted

        Args:
            file (h5py.File): Opened file to load from
        """
        try:
            self.stain_matrix_target = file[self.stain_matrix_key][()]
        except ValueError:
            print(
                f"Invalid stain matrix keys. Expected key is "
                f"{self.stain_matrix_key}."
            )

    def _save_values(self, file: h5py.File) -> None:
        """Save values needed to reproduce fit

        Args:
            file (h5py.File): Opened file to save to
        """
        file.create_dataset(
            self.stain_matrix_key,
            data=self.stain_matrix_target,
            compression="gzip",
            compression_opts=9,
        )

    @staticmethod
    def _notwhite_mask(image: np.ndarray, threshold: float) -> np.ndarray:
        """Computed a mask where the image has values over the percentage threshold
           in LAB color space

        Args:
            image (np.array): Image to compute the mask for
            threshold (float): Threshold in percent, 0 <= threshold <= 1

        Returns:
            np.array: Mask for the image
        """
        image_lab = rgb2lab(image)
        lightness = image_lab[:, :, 0] / 100.0
        return lightness < threshold

    def _get_stain_matrix(self, input_image: np.ndarray) -> np.ndarray:
        """Compute the 2x3 stain matrix with the method from the paper

        Args:
            input_image (np.array): Image to extract the stains from

        Returns:
            np.array: Extracted stains
        """
        mask = self._notwhite_mask(
            input_image, threshold=self.thres).reshape(
            (-1,))
        optical_density = self._rgb_to_od(input_image).reshape((-1, 3))
        optical_density = optical_density[mask]
        n_features = optical_density.T.shape[1]

        dict_learner = DictionaryLearning(
            n_components=2,
            alpha=self.lambda_s,
            max_iter=10,
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
            transform_n_nonzero_coefs=n_features,
            random_state=0,
            positive_dict=True,
        )
        dictionary = dict_learner.fit_transform(optical_density.T).T

        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]
        dictionary = self._normalize_rows(dictionary)
        return dictionary

    def _normalize_image(self, input_image: np.ndarray) -> np.ndarray:
        """Compute the normalization according to the paper

        Args:
            input_image (np.array): Image to normalize

        Returns:
            np.array: Normalized image
        """
        stain_matrix_source = self._get_stain_matrix(input_image)
        source_concentrations = self._get_concentrations(
            input_image, stain_matrix_source
        )
        return (255 * np.exp(-1 * np.dot(source_concentrations,
                                         self.stain_matrix_target).reshape(input_image.shape))).astype(np.uint8)

    def fit(self, target_image: np.ndarray) -> None:
        """Fit the normalizer to a target value and save it for the future

        Args:
            target_image (np.array): Target image
        """
        target_image = self._standardize_brightness(target_image)
        self.stain_matrix_target = self._get_stain_matrix(target_image)
        self._save_precomputed()
