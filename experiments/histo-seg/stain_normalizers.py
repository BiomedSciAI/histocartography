from abc import abstractmethod
import logging

from skimage.color import rgb2lab
import h5py
import numpy as np
import spams
from PIL import Image

from utils import PipelineStep


class StainNormalizer(PipelineStep):
    """Base class for creating fancy stain normalizers"""

    def __init__(self, lambda_c: float = 0.01, **kwargs) -> None:
        """Create a stain normalizer with appropriate penaltz for optimizing getting stains

        Args:
            lambda_c (float, optional): lambda parameter for getting the concentration. Defaults to 0.01.
        """
        self.lambda_c = lambda_c
        super().__init__(**kwargs)
        self.save_path = self.output_dir / "normalizer.h5"

    @staticmethod
    def _standardize_brightness(input_image: np.ndarray) -> np.array:
        """Standardize an image by moving all values above the 90 percentile to 255

        Args:
            input_image (np.array): Image to normalize

        Returns:
            np.array: Normalized image
        """
        upper_percentile_bound = np.percentile(input_image, 90)
        return np.clip(input_image * 255.0 / upper_percentile_bound, 0, 255).astype(
            np.uint8
        )

    @staticmethod
    def _RGB_to_OD(input_image: np.ndarray) -> np.array:
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
    def _normalize_rows(input_array: np.ndarray) -> np.array:
        """Normalizes the rows of a given image

        Args:
            input_array (np.array): Image to normalize

        Returns:
            np.array: Normalized image
        """
        return input_array / np.linalg.norm(input_array, axis=1)[:, None]

    def _get_concentrations(
        self, input_image: np.ndarray, stain_matrix: np.ndarray
    ) -> np.array:
        """Extracts the stain concentrations for all pixels of the given input image

        Args:
            input_image (np.array): Input image
            stain_matrix (np.array): Compute stain vectors in the OD color space

        Returns:
            np.array: Extracted stains of all pixels (vectorized image)
        """
        OD = self._RGB_to_OD(input_image).reshape((-1, 3))
        return (
            spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=self.lambda_c, pos=True, numThreads=1)
            .toarray()
            .T
        )

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
        assert (
            self.base_path is not None
        ), f"Can only save intermediate output if base_path was not None when constructing the object"
        input_file = h5py.File(self.save_path, "r")
        self._load_values(input_file)
        input_file.close()

    def _save_precomputed(self):
        """Saves the precomputed values"""
        assert (
            self.base_path is not None
        ), "Can only save intermediate output if base_path was not None when constructing the object"
        output_file = h5py.File(self.save_path, "w")
        self._save_values(output_file)
        output_file.close()

    @abstractmethod
    def _normalize_image(self, input_image: np.ndarray) -> np.array:
        """Perform the normalization of an image with precomputed values

        Args:
            input_image (np.array): Image to normalize

        Returns:
            np.array: Normalized image
        """

    def process(self, input_image: np.ndarray) -> np.array:
        """Stain normalizes a given image

        Args:
            input_image (np.array): Input image

        Returns:
            np.array: The stain normalized image
        """
        self._load_precomputed()
        standardized_image = self._standardize_brightness(input_image)
        normalized_image = self._normalize_image(standardized_image)
        return normalized_image

    def process_and_save(self, output_name: str, **kwargs) -> np.array:
        """Process and save in the provided path as a png image

        Args:
            output_name (str): Name of output file
        """
        assert (
            self.base_path is not None
        ), f"Can only save intermediate output if base_path was not None when constructing the object"
        output_path = self.output_dir / f"{output_name}.png"
        if output_path.exists():
            logging.info(
                f"{self.__class__.__name__}: Output of {output_name} already exists, using it instead of recomputing"
            )
            input_file = Image.open(output_path)
            output = np.array(input_file)
            input_file.close()
        else:
            output = self.process(**kwargs)
            output_image = Image.fromarray(output)
            output_image.save(output_path)
            output_image.close()
        return output


class MacenkoStainNormalizer(StainNormalizer):
    """
    Stain normalization based on the method of:
    M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’,
    in 2009 IEEE International Symposium on Biomedical Imaging:
    From Nano to Macro, 2009, pp. 1107–1110.
    """

    def __init__(
        self, target: str, alpha: float = 1.0, beta: float = 0.15, **kwargs
    ) -> None:
        """Apply the stain normalization with a given target and parameters

        Args:
            target (str): Name of the target image for identification
            alpha (float, optional): Alpha parameter. Defaults to 1.0.
            beta (float, optional): Beta parameter. Defaults to 0.15.
        """
        self.alpha = alpha
        self.beta = beta
        self.target = target
        super().__init__(**kwargs)
        # Hidden fields
        self.stain_matrix_key = "stain_matrix"
        self.max_C_target_key = "maxC_target"
        self.stain_matrix_target = None
        self.max_C_target = None

    def _load_values(self, file: h5py.File) -> None:
        """Loads the values computed when fitted

        Args:
            file (h5py.File): Opened file to load from
        """
        self.stain_matrix_target = file[self.stain_matrix_key][()]
        self.max_C_target = file[self.max_C_target_key][()]

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
            self.max_C_target_key,
            data=self.max_C_target,
            compression="gzip",
            compression_opts=9,
        )

    def _get_stain_matrix(self, input_image: np.ndarray) -> np.array:
        """Compute the 2x3 stain matrix with the method from the paper

        Args:
            input_image (np.array): Image to extract the stains from

        Returns:
            np.array: Extracted stains
        """
        OD = self._RGB_to_OD(input_image).reshape((-1, 3))
        OD = OD[(OD > self.beta).any(axis=1), :]
        _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
        V = V[:, [2, 1]]
        if V[0, 0] < 0:
            V[:, 0] *= -1
        if V[0, 1] < 0:
            V[:, 1] *= -1
        That = np.dot(OD, V)
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, self.alpha)
        maxPhi = np.percentile(phi, 100 - self.alpha)
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        if v1[0] > v2[0]:
            HE = np.array([v1, v2])
        else:
            HE = np.array([v2, v1])
        return self._normalize_rows(HE)

    def _normalize_image(self, input_image: np.ndarray) -> np.array:
        """Compute the normalization according to the paper

        Args:
            input_image (np.array): Image to normalize

        Returns:
            np.array: Normalized image
        """
        self._load_precomputed()
        stain_matrix_source = self._get_stain_matrix(input_image)
        source_concentrations = self._get_concentrations(
            input_image, stain_matrix_source
        )
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        maxC_target = self.max_C_target
        source_concentrations *= maxC_target / maxC_source
        return (
            255
            * np.exp(
                -1
                * np.dot(source_concentrations, self.stain_matrix_target).reshape(
                    input_image.shape
                )
            )
        ).astype(np.uint8)

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
        self.max_C_target = np.percentile(target_concentrations, 99, axis=0).reshape(
            (1, 2)
        )
        self._save_precomputed()


class VahadaneStainNormalizer(StainNormalizer):
    """
    Stain normalization inspired by method of:
    A. Vahadane et al., ‘Structure-Preserving Color Normalization and Sparse Stain Separation
    for Histological Images’, IEEE Transactions on Medical Imaging,
    vol. 35, no. 8, pp. 1962–1971, Aug. 2016.
    """

    def __init__(self, target, threshold=0.8, lambda_s=0.1, **kwargs) -> None:
        self.target = target
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
        self.stain_matrix_target = file[self.stain_matrix_key][()]

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
    def _notwhite_mask(image: np.ndarray, threshold: float) -> np.array:
        """Computed a mask where the image has values over the percentage threshold in LAB color space

        Args:
            image (np.array): Image to compute the mask for
            threshold (float): Threshold in percent, 0 <= threshold <= 1

        Returns:
            np.array: Mask for the image
        """
        image_lab = rgb2lab(image)
        lightness = image_lab[:, :, 0] / 255.0
        return lightness < threshold

    def _get_stain_matrix(self, input_image: np.ndarray) -> np.array:
        """Compute the 2x3 stain matrix with the method from the paper

        Args:
            input_image (np.array): Image to extract the stains from

        Returns:
            np.array: Extracted stains
        """
        mask = self._notwhite_mask(input_image, threshold=self.thres).reshape((-1,))
        OD = self._RGB_to_OD(input_image).reshape((-1, 3))
        OD = OD[mask]
        dictionary = spams.trainDL(
            OD.T,
            K=2,
            lambda1=self.lambda_s,
            mode=2,
            modeD=0,
            posAlpha=True,
            posD=True,
            verbose=False,
            numThreads=1,
        ).T
        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]
        dictionary = self._normalize_rows(dictionary)
        return dictionary

    def _normalize_image(self, input_image: np.ndarray) -> np.array:
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
        return (
            255
            * np.exp(
                -1
                * np.dot(source_concentrations, self.stain_matrix_target).reshape(
                    input_image.shape
                )
            )
        ).astype(np.uint8)

    def fit(self, target_image: np.ndarray) -> None:
        """Fit the normalizer to a target value and save it for the future

        Args:
            target_image (np.array): Target image
        """
        target_image = self._standardize_brightness(target_image)
        self.stain_matrix_target = self._get_stain_matrix(target_image)
        self._save_precomputed()
