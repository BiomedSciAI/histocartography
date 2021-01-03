import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from histocartography.preprocessing.pipeline import BatchPipelineRunner
from histocartography.utils import dynamic_import_from

from utils import merge_metadata, start_logging


def preprocessing(
    config: dict,
    dataset: str,
    cores: int = 1,
    labels: bool = True,
    save: bool = True,
    test: bool = False,
    link_directory: Optional[str] = None,
):
    PREPROCESS_PATH = dynamic_import_from(dataset, "PREPROCESS_PATH")
    IMAGES_DF = dynamic_import_from(dataset, "IMAGES_DF")
    metadata = pd.read_pickle(IMAGES_DF)
    if labels:
        ANNOTATIONS_DF = dynamic_import_from(dataset, "ANNOTATIONS_DF")
        annotation_metadata = pd.read_pickle(ANNOTATIONS_DF)
        metadata = merge_metadata(metadata, annotation_metadata)
    if save and not PREPROCESS_PATH.exists():
        PREPROCESS_PATH.mkdir()
    if test:
        metadata = metadata.iloc[[0]]
        cores = 1
        config["stages"][3]["superpixel"]["params"]["nr_superpixels"] = 300
        config["stages"][1]["stain_normalizers"]["params"][
            "target_path"
        ] = metadata.iloc[0].path
    else:
        # Inject target path into config
        target = config["stages"][1]["stain_normalizers"]["params"]["target"]
        target_path = metadata.loc[target, "path"]
        config["stages"][1]["stain_normalizers"]["params"]["target_path"] = target_path

    pipeline = BatchPipelineRunner(
        output_path=PREPROCESS_PATH, pipeline_config=config, save=save
    )
    pipeline.run(metadata=metadata, cores=cores)
    if link_directory is not None:
        pipeline.link_output(link_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/preprocess.yml")
    parser.add_argument("--level", type=str, default="WARNING")
    parser.add_argument("--test", action="store_const", const=True, default=False)
    args = parser.parse_args()

    start_logging(args.level)
    assert Path(args.config).exists(), f"Config path does not exist: {args.config}"
    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    logging.info("Start preprocessing")
    assert (
        "pipeline" in config
    ), f"pipeline not defined in config {args.config}: {config.keys()}"
    assert (
        "params" in config
    ), f"params not defined in config {args.config}: {config.keys()}"
    preprocessing(
        test=args.test,
        config=config["pipeline"],
        **config["params"],
    )
