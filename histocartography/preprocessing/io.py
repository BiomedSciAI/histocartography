from pathlib import Path
from typing import Any, Union
import h5py

import dgl
import numpy as np
from dgl.data.utils import load_graphs

from ..pipeline import PipelineStep
from .utils import load_image
from ..utils.io import h5_to_numpy 


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
    # type: ignore[override]
    def _process(self, path: Union[str, Path]) -> np.ndarray:
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


class H5Loader(FileLoader):
    def _process(  # type: ignore[override]
        self, path: Union[str, Path]
    ) -> Any:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            if len(keys) == 1:
                return h5_to_numpy(f[keys[0]])
            else:
                out = tuple([h5_to_numpy(f[key]) for key in keys])
                return out
