import io
from abc import abstractmethod
from pathlib import Path
from typing import Any

import dgl
import numpy as np
import pandas as pd
from networkx.algorithms.distance_measures import diameter

from .pipeline import PipelineStep


class StatsComputer(PipelineStep):
    def mkdir(self) -> Path:
        """Create path to output files"""
        assert (
            self.base_path is not None
        ), "Can only create directory if base_path was not None when constructing the object"
        return self.base_path

    @property
    def _file_ending(self):
        return ".csv"

    @property
    def _separator(self):
        return ","

    @property
    def _header(self):
        """Returns csv header"""
        return "name" + self._separator + "value" + "\n"

    @property
    def _file_path(self):
        return str(Path(self.base_path) / f"{self._filename}{self._file_ending}")

    @property
    @abstractmethod
    def _filename(self):
        """Output filename"""

    def _write(self, name, value):
        line = f"{name}{self._separator}{value}\n"
        with io.open(self._file_path, mode="a") as file:
            file.write(line)

    def precompute(self, final_path) -> None:
        if self.base_path is not None:
            with open(self._file_path, mode="w") as file:
                file.write(self._header)
        return super().precompute(final_path)

    def process_and_save(self, output_name, *args, **kwargs: Any) -> Any:
        value = self.process(*args, **kwargs)
        self._write(output_name, value)


class GraphDiameter(StatsComputer):
    @property
    def _filename(self):
        """Output filename"""
        return "graph_diameter"

    def process(self, graph: dgl.DGLGraph) -> int:
        networkx_graph = graph.to_networkx()
        return diameter(networkx_graph)


class SuperpixelCounter(StatsComputer):
    @property
    def _filename(self):
        """Output filename"""
        return "nr_superpixels"

    def process(self, superpixels: np.ndarray) -> int:
        return len(pd.unique(np.ravel(superpixels)))
