import sys


# Fake it till you make it: fake SimpleITK dependency to avoid installing ITK on the cluster.
class SimpleITK(object):
    def __getattr__(self, name):
        pass


sys.modules["SimpleITK"] = SimpleITK()

import cv2
import numpy as np
import torch
import torchio as tio
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import SegmentationFromCNN
from histocartography.interpretability.saliency_explainer.graph_gradcam_explainer import (
    GraphGradCAMExplainer,
)
from utils import get_segmentation_map
import dgl


class BaseInference:
    def __init__(self, model, device=None) -> None:
        super().__init__()
        self.model = model.eval()
        if device is not None:
            self.device = device
        else:
            self.device = next(model.parameters()).device
        self.model = self.model.to(self.device)


class PatchBasedInference(BaseInference):
    def __init__(
        self,
        model,
        patch_size,
        overlap,
        batch_size,
        nr_classes,
        num_workers,
        device=None,
    ) -> None:
        super().__init__(model=model, device=device)
        assert len(patch_size) == 2
        assert len(overlap) == 2
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.overlap = overlap
        self.num_workers = num_workers

    def _predict(self, subject: tio.Subject, operation: str):
        if operation == "per_class":
            output_channels = self.nr_classes
        elif operation == "argmax":
            output_channels = 1
        else:
            raise NotImplementedError(
                f"Only support operation [per_class, argmax], but got {operation}"
            )

        _, height, width = subject.spatial_shape
        label = tio.Image(
            tensor=torch.zeros((1, output_channels, height, width)), type=tio.LabelMap
        )  # Fake tensor to create subject
        output_subject = tio.Subject(label=label)  # Fake subject to create sampler

        image_sampler = tio.inference.GridSampler(
            subject=subject,
            patch_size=(3, self.patch_size[0], self.patch_size[1]),
            patch_overlap=(0, self.overlap[0], self.overlap[1]),
        )
        label_sampler = tio.inference.GridSampler(
            subject=output_subject,
            patch_size=(1, self.patch_size[0], self.patch_size[1]),
            patch_overlap=(0, self.overlap[0], self.overlap[1]),
        )  # Fake sampler to create aggregator
        aggregator = tio.inference.GridAggregator(label_sampler, overlap_mode="average")
        image_loader = DataLoader(
            image_sampler, batch_size=self.batch_size, num_workers=self.num_workers
        )

        with torch.no_grad():
            for batch in tqdm(image_loader, leave=False):
                input_tensor = (
                    batch["image"][tio.DATA].squeeze(1).to(self.device)
                )  # Remove redundant dimension added for tio
                locations = torch.as_tensor(batch[tio.LOCATION])
                locations[:, 3] = output_channels  # Fix output size
                logits = self.model(input_tensor)
                soft_predictions = logits.sigmoid()

                if operation == "per_class":
                    output_tensor = (
                        soft_predictions.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, self.patch_size[0], self.patch_size[1])
                    )  # Repeat output to size of image
                elif operation == "argmax":
                    hard_predictions = soft_predictions.argmax(dim=1).to(torch.int32)
                    output_tensor = (
                        hard_predictions.unsqueeze(1)
                        .unsqueeze(2)
                        .repeat(1, self.patch_size[0], self.patch_size[1])
                    )  # Repeat output to size of image
                else:
                    raise NotImplementedError(
                        f"Only support operation [per_class, argmax], but got {operation}"
                    )
                aggregator.add_batch(output_tensor.unsqueeze(1), locations)
        if operation == "per_class":
            return aggregator.get_output_tensor()[0].detach().cpu().numpy()
        elif operation == "argmax":
            return aggregator.get_output_tensor()[0][0].cpu().cpu().numpy()
        else:
            raise NotImplementedError(
                f"Only support operation [per_class, argmax], but got {operation}"
            )

    def predict(self, image: torch.Tensor, operation="per_class"):
        input_tio_image = tio.Image(tensor=image.unsqueeze(0), type=tio.INTENSITY)
        subject = tio.Subject(image=input_tio_image)
        return self._predict(subject, operation)


class GraphNodeBasedInference(BaseInference):
    def __init__(self, model, device, NR_CLASSES) -> None:
        super().__init__(model, device=device)
        self.NR_CLASSES = NR_CLASSES

    def predict(self, graph, superpixels, operation="argmax"):
        assert operation == "argmax"

        graph = graph.to(self.device)
        node_logits = self.model(graph)
        if isinstance(node_logits, tuple):
            node_logits = node_logits[1]

        node_predictions = node_logits.argmax(axis=1).detach().cpu().numpy()
        segmentation_maps = get_segmentation_map(
            node_predictions=node_predictions,
            superpixels=superpixels,
            NR_CLASSES=self.NR_CLASSES,
        )
        return segmentation_maps


class GraphGradCAMBasedInference(BaseInference):
    def __init__(self, NR_CLASSES, model, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.NR_CLASSES = NR_CLASSES
        self.explainer = GraphGradCAMExplainer(model=model)

    def predict(self, graph, superpixels, operation="argmax"):
        assert operation == "argmax"

        graph = graph.to(self.device)
        importances, logits = self.explainer.process_all(
            graph, list(range(self.NR_CLASSES))
        )
        node_importances = (
            importances * torch.as_tensor(logits)[0].sigmoid().numpy()[:, np.newaxis]
        ).argmax(0)
        return get_segmentation_map(node_importances, superpixels, self.NR_CLASSES)

    def predict_batch(self, graph, superpixels, operation="argmax"):
        segmentation_maps = list()
        for i, graph in enumerate(dgl.unbatch(graph)):
            segmentation_map = self.predict(graph, superpixels[i], operation)
            segmentation_maps.append(segmentation_map)
        return np.stack(segmentation_maps)


class ImageInferenceModel(BaseInference):
    def __init__(self, model, device, final_shape) -> None:
        segmentation_model = SegmentationFromCNN(model)
        super().__init__(segmentation_model, device=device)
        self.final_shape = final_shape

    def predict(self, input_tensor, operation: str):
        input_tensor = input_tensor.to(self.device)
        logits = self.model(input_tensor.unsqueeze(0))
        soft_predictions = logits.sigmoid()
        if operation == "per_class":
            predictions = soft_predictions[0].detach().cpu()
        elif operation == "argmax":
            hard_predictions = soft_predictions.argmax(dim=1)
            predictions = hard_predictions[0].detach().cpu()
        else:
            raise NotImplementedError(
                f"Only support operation [per_class, argmax], but got {operation}"
            )
        if self.final_shape is None:
            return predictions.numpy()
        return cv2.resize(
            predictions.numpy(),
            dsize=self.final_shape,
            interpolation=cv2.INTER_NEAREST,
        )
