from pathlib import Path
from typing import Any, Union

import dgl
import numpy as np
from dgl.data.utils import load_graphs

from ..pipeline import PipelineStep
from .utils import load_image


class FileLoader(PipelineStep):
    def mkdir(self) -> Path:
        """Create path to output files"""
        assert (
            self.save_path is not None
        ), "Can only create directory if base_path was not None when constructing the object"
        return Path(self.save_path)

    def _process_and_save(self, *args, output_name, **kwargs: Any) -> Any:
        return self._process(*args, **kwargs)


class ImageLoader(FileLoader):
    def _process(self, path: Union[str, Path]) -> np.ndarray:  # type: ignore[override]
        image_path = Path(path)
        image = load_image(image_path)
        return image


class DGLGraphLoader(FileLoader):
    def _process(  # type: ignore[override]
        self, path: Union[str, Path]
    ) -> dgl.DGLGraph:
        graph_path = str(path)  # DGL cannot handle pathlib.Path
        graphs, _ = load_graphs(graph_path)
        if len(graphs) == 1:
            return graphs[0]
        return graphs
