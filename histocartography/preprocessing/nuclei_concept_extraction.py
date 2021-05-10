"""Extract features from images for a given structure"""

import numpy as np
import torch

from ..pipeline import PipelineStep
from .feature_extraction import HANDCRAFTED_FEATURES_NAMES, HandcraftedFeatureExtractor


class NucleiConceptExtractor(PipelineStep):
    """Class for Nuclei concept extraction.
    Extract nuclei-level measurable concepts.
    """

    def __init__(self, concept_names=None, **kwargs) -> None:
        """Nuclei Concept Extractor constructor.

        Args:
            concept_names (str): List of all the concepts to extract. Default to ''.
                                 If set to None, extract all the concepts.
                                 Otherwise, extract all the listed concepts
                                separated with commas, eg. 'area,perimeter,eccentricity'.
        """
        super().__init__(**kwargs)

        if concept_names is not None:
            self.concept_names = concept_names.split(",")
        else:
            self.concept_names = concept_names
        self.hc_feature_extractor = HandcraftedFeatureExtractor()

    def _process(  # type: ignore[override]
        self, input_image: np.ndarray, instance_map: np.ndarray
    ) -> torch.Tensor:
        """Extract nuclei-level concepts from the input_image for
           the detected nuclei defined in instance_map

        Args:
            input_image (np.array): Original RGB image
            instance_map (np.array): Extracted instance_map

        Returns:
            np.ndarray: nuclei concept
        """

        nuclei_concepts = self.hc_feature_extractor.process(input_image, instance_map)

        if self.concept_names is not None:
            indices_to_keep = [
                HANDCRAFTED_FEATURES_NAMES[c] for c in self.concept_names
            ]
            nuclei_concepts = nuclei_concepts.index_select(
                1, torch.LongTensor(indices_to_keep)
            )

        # convert to numpy array
        nuclei_concepts = nuclei_concepts.cpu().detach().numpy()

        return nuclei_concepts
