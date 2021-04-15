import importlib
from typing import Any, Iterable, Tuple

from .io import download_example_data, download_test_data
from .io import download_box_link, is_box_url
from .graph import set_graph_on_cuda, set_graph_on_cpu

__all__ = [
    'download_example_data',
    'download_test_data',
    'download_box_link',
    'set_graph_on_cuda',
    'set_graph_on_cpu',
    'is_box_url'
]


def dynamic_import_from(source_file: str, class_name: str) -> Any:
    """Do a from source_file import class_name dynamically

    Args:
        source_file (str): Where to import from
        class_name (str): What to import

    Returns:
        Any: The class to be imported
    """
    module = importlib.import_module(source_file)
    return getattr(module, class_name)


def signal_last(input_iterable: Iterable[Any]) -> Iterable[Tuple[bool, Any]]:
    iterable = iter(input_iterable)
    return_value = next(iterable)
    for value in iterable:
        yield False, return_value
        return_value = value
    yield True, return_value
