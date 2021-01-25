from pathlib import Path

from numpy.lib.function_base import gradient
from .pipeline import PipelineStep
from typing import Union, Any
from PIL import Image
import logging
import numpy as np
import cv2


class TissueMask(PipelineStep):
    def mkdir(self) -> Path:
        """Create path to output files"""
        assert (
            self.base_path is not None
        ), "Can only create directory if base_path was not None when constructing the object"
        if not self.output_dir.exists():
            self.output_dir.mkdir()
        return self.base_path

    def process_and_save(self, output_name: str, *args, **kwargs) -> np.ndarray:
        """Process and save in the provided path as a png image

        Args:
            output_name (str): Name of output file
        """
        assert (
            self.base_path is not None
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
            output = self.process(*args, **kwargs)
            with Image.fromarray(output) as output_image:
                output_image.save(output_path)
        return output

    def precompute(self, final_path) -> None:
        """Precompute all necessary information"""
        if self.base_path is not None:
            self._link_to_path(Path(final_path) / "tissue_masks")


class BrightnessThresholdTissueMask(TissueMask):
    def __init__(
        self,
        base_path: Union[None, str, Path],
        blur_size=25,
        dilation_steps=5,
        erosion_steps=10,
        kernel_size=20,
    ) -> None:
        self.blur_size = blur_size
        self.dilation_steps = dilation_steps
        self.erosion_steps = erosion_steps
        self.kernel_size = kernel_size
        super().__init__(base_path=base_path)
        self.kernel = np.ones((self.kernel_size, self.kernel_size), "uint8")

    def process(self, image) -> Any:
        """According to https://github.com/eiriniar/gleason_CNN/blob/master/utils/create_tissue_masks.py"""
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(grey_image, (self.blur_size, self.blur_size), 0)
        ret, img_thres = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # add padding to avoid weird borders afterwards
        bb = self.dilation_steps * 2 + self.erosion_steps + 2 * self.kernel_size + 40
        img_thres = cv2.copyMakeBorder(
            img_thres, bb, bb, bb, bb, cv2.BORDER_CONSTANT, value=0
        )
        # dilation to fill black holes
        img = cv2.dilate(img_thres, self.kernel, iterations=self.dilation_steps)
        # followed by erosion to restore borders, eat up small objects
        img = cv2.erode(img, self.kernel, iterations=self.erosion_steps)
        # then dilate again
        img = cv2.dilate(img, self.kernel, iterations=self.dilation_steps)
        # crop to restore original image
        ws = np.array(img)[bb:-bb, bb:-bb]
        return ws
