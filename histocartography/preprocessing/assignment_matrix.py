"""This module handles the cell-to-tissue assignment"""

import logging
from pathlib import Path
import pandas as pd

import numpy as np
from ..pipeline import PipelineStep


class AssignmnentMatrixBuilder(PipelineStep):
    """
    Assigning low-level instances to high-level instances using instance maps.
    """

    def _process(
        self, low_level_centroids: np.ndarray, high_level_map: np.ndarray
    ) -> np.ndarray:
        """Construct assignment between low-level and high-level instances
        Args:
            low_level_centroids (np.array): Extracted instance centroids in low-level
            high_level_map (np.array): Extracted high-level instance map
        Returns:
            np.ndarray: Constructed assignment
        """
        return self._build_assignment_matrix(
            low_level_centroids, high_level_map)

    def _build_assignment_matrix(
            self, low_level_centroids: np.ndarray, high_level_map: np.ndarray
    ) -> np.ndarray:
        """Construct assignment between inter-level instances"""
        low_level_centroids = low_level_centroids.astype(int)
        high_instance_ids = np.sort(
            pd.unique(np.ravel(high_level_map))).astype(int)
        if 0 in high_instance_ids:
            high_instance_ids = np.delete(
                high_instance_ids, np.where(
                    high_instance_ids == 0))

        low_to_high = high_level_map[
            low_level_centroids[:, 1],
            low_level_centroids[:, 0]
        ].astype(int)

        assignment_matrix = np.zeros(
            (
                low_level_centroids.shape[0],
                len(high_instance_ids)
            )
        )
        # relevant instances in high_level_map begins from id=1
        assignment_matrix[np.arange(low_to_high.size), low_to_high - 1] = 1
        return assignment_matrix
