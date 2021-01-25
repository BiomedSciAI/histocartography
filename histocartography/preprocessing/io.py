from pathlib import Path
from typing import Any

from .pipeline import PipelineStep
from .utils import load_image


class FileLoader(PipelineStep):
    def mkdir(self) -> Path:
        """Create path to output files"""
        assert (
            self.base_path is not None
        ), "Can only create directory if base_path was not None when constructing the object"
        return self.base_path

    def process_and_save(self, output_name, *args, **kwargs: Any) -> Any:
        return self.process(*args, **kwargs)


class ImageLoader(FileLoader):
    def process(self, path: str, *args, **kwargs) -> Any:
        image_path = Path(path)
        image = load_image(image_path)
        return image

class DGLGraphLoader(FileLoader):
    def process(self, path: str, *args, **kwargs) -> Any:
        graph_path = Path(path)
        graphs, _ = load_image(graph_path)
        if len(graphs) == 1:
            return graphs[0]
        return graphs
