import io
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import dgl
import numpy as np
import pandas as pd
from networkx.algorithms.distance_measures import diameter

from ..pipeline import PipelineStep


class StatsComputer(PipelineStep):
    def mkdir(self) -> Path:
        """Create path to output files"""
        assert (
            self.save_path is not None
        ), "Can only create directory if base_path was not None when constructing the object"
        return Path(self.save_path)

    @property
    def _file_ending(self) -> str:
        """File ending to save to

        Returns:
            str: File ending
        """
        return ".csv"

    @property
    def _separator(self) -> str:
        """String separator

        Returns:
            str: Symbol used to seperate name and entry
        """
        return ","

    @property
    def _header(self) -> str:
        """File header

        Returns:
            str: File header
        """
        return "name" + self._separator + "value" + "\n"

    @property
    @abstractmethod
    def _filename(self):
        """Output filename"""

    def _write(self, name: str, value: Any):
        """Writes a single line to the file. When running on Unix the append mode is
           atomic as long as the string does not exceed the buffer.

        Args:
            name (str): Unique identifier of datapoint
            value (Any): Computed value for datapoint
        """
        assert self._file_path is not None, f"Precompute not called before writing."
        line = f"{name}{self._separator}{value}\n"
        with io.open(self._file_path, mode="a") as file:
            file.write(line)

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Creates the file to write to and write the file header

        Args:
            link_path (Union[None, str, Path], optional): Ignored. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to the file header.
                If None is provided it uses self.save_path. Defaults to None.
        """
        self._file_path: Optional[str]
        if self.save_path is not None:
            if precompute_path is None:
                precompute_path = self.save_path
            self._file_path = str(
                Path(precompute_path) / f"{self._filename}{self._file_ending}"
            )
            with open(self._file_path, mode="w") as file:
                file.write(self._header)
        else:
            self._file_path = None

    def _process_and_save(
        self, *args: Any, output_name: str, **kwargs: Any
    ) -> Any:
        """Compute the value and write it to the file and return it

        Args:
            output_name (str): Unique identifier of datapoint

        Returns:
            Any: Computed value
        """
        value = self._process(*args, **kwargs)
        self._write(output_name, value)
        return value


class GraphDiameter(StatsComputer):
    @property
    def _filename(self) -> str:
        """Output filename

        Returns:
            str: Output filename
        """
        return "graph_diameter"

    def _process(self, graph: dgl.DGLGraph) -> int:  # type: ignore[override]
        """Compute the graph diameter

        Args:
            graph (dgl.DGLGraph): Input graph

        Returns:
            int: Diameter of graph
        """
        networkx_graph = graph.to_networkx()
        return diameter(networkx_graph)


class SuperpixelCounter(StatsComputer):
    @property
    def _filename(self) -> str:
        """Output filename

        Returns:
            str: Output filename
        """
        return "nr_superpixels"

    # type: ignore[override]
    def _process(self, superpixels: np.ndarray) -> int:
        """Compute the number of superpixels

        Args:
            superpixels (np.ndarray): Input instance map

        Returns:
            int: Number of superpixels
        """
        return len(pd.unique(np.ravel(superpixels)))
