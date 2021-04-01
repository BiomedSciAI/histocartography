"""This module handles the cell-to-tissue assignment"""

import logging
from abc import abstractmethod
from pathlib import Path

import numpy as np
from ..pipeline import PipelineStep


class BaseMatrixBuilder(PipelineStep):
    """
    Base interface class for assignment matrix.
    """
    def process(
        self, instance_centroids: np.ndarray, instance_map: np.ndarray
    ) -> np.ndarray:
        """Construct assignment between low-level and high-level instances
        Args:
            instance_centroids (np.array): Extracted low-level instance centroids
            instance_map (np.array): Extracted high-level instance map
        Returns:
            np.ndarray: Extracted assignment
        """
        return self._build_assignment_matrix(instance_centroids, instance_map)

    @abstractmethod
    def _build_assignment_matrix(
        self, instance_centroids: np.ndarray, instance_map: np.ndarray
    ) -> np.ndarray:
        """Construct assignment between inter-level instances
        Args:
            instance_centroids (np.array): Extracted low-level instance centroids
            instance_map (np.array): Extracted high-level instance map
        Returns:
            np.ndarray: Extracted assignment
        """

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


class AssignmnentMatrixBuilder(BaseMatrixBuilder):
    """
    Assigning low-level instances to high-level instances
    """

    def __init__(self, **kwargs) -> None:
        """Construct assignment between low-level and high-level instances"""
        logging.debug("*** Assignment Matrix ***")
        super().__init__(**kwargs)

    def _build_assignment_matrix(
            self, instance_centroids: np.ndarray, instance_map: np.ndarray
    ) -> np.ndarray:
        """Construct assignment between inter-level instances
        Args:
            instance_centroids (np.array): Extracted low-level instance centroids
            instance_map (np.array): Extracted high-level instance map
        Returns:
            np.ndarray: Extracted assignment
        """
        instance_centroids = instance_centroids.astype(int)
        low_to_high = instance_map[
            instance_centroids[:, 1],
            instance_centroids[:, 0]
        ].astype(int)

        assignment_matrix = np.zeros(
            (
                instance_centroids.shape[0],
                len(np.unique(instance_map))
            )
        )
        assignment_matrix[np.arange(low_to_high.size), low_to_high] = 1
        return assignment_matrix