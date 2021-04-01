"""This module handles the cell-to-tissue assignment"""

import logging
from pathlib import Path
import pandas as pd

import numpy as np
from ..pipeline import PipelineStep


class AssignmnentMatrixBuilder(PipelineStep):
    """
    Assigning low-level instances to high-level instances
    """
    def process(
        self, low_level_centroids: np.ndarray, high_level_map: np.ndarray
    ) -> np.ndarray:
        """Construct assignment between low-level and high-level instances
        Args:
            low_level_centroids (np.array): Extracted instance centroids in low-level
            high_level_map (np.array): Extracted high-level instance map
        Returns:
            np.ndarray: Constructed assignment
        """
        return self._build_assignment_matrix(low_level_centroids, high_level_map)

    def _build_assignment_matrix(
            self, low_level_centroids: np.ndarray, high_level_map: np.ndarray
    ) -> np.ndarray:
        """Construct assignment between inter-level instances"""
        low_level_centroids = low_level_centroids.astype(int)
        high_instance_ids = np.sort(pd.unique(np.ravel(high_level_map))).astype(int)
        if 0 in high_instance_ids:
            high_instance_ids = np.delete(high_instance_ids, np.where(high_instance_ids == 0))

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

    def process_and_save(self, output_name: str, *args, **kwargs) -> np.ndarray:
        """Process and save in the provided path as a npy file
        Args:
            output_name (str): Name of output file
        """
        assert (
            self.base_path is not None
        ), "Can only save intermediate output if base_path was not None during construction"
        output_path = self.output_dir / f"{output_name}.npy"
        if output_path.exists():
            logging.info(
                "%s: Output of %s already exists, using it instead of recomputing",
                self.__class__.__name__,
                output_name,
            )
            try:
                output = np.load(output_path)
            except OSError as error:
                logging.critical("Could not open %s", output_path)
                raise error
        else:
            output = self.process(*args, **kwargs)
            np.save(output_path, output)
        return output

    def precompute(self, final_path) -> None:
        """Precompute all necessary information"""
        if self.base_path is not None:
            self._link_to_path(Path(final_path) / "assignment_matrix")