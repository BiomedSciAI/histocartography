import sys


# Fake it till you make it: fake SimpleITK dependency to avoid installing ITK on the cluster.
class SimpleITK(object):
    def __getattr__(self, name):
        pass


sys.modules["SimpleITK"] = SimpleITK()

import torch
import torchio as tio
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class PatchBasedInference:
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
        super().__init__()
        assert len(patch_size) == 2
        assert len(overlap) == 2
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.overlap = overlap
        self.model = model.eval()
        if device is not None:
            self.device = device
        else:
            self.device = next(model.parameters()).device
        self.nr_classes = nr_classes
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
            tensor=torch.zeros((1, output_channels, height, width))
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
                    hard_predictions = soft_predictions.argmax(dim=1)
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
            return aggregator.get_output_tensor()[0].cpu()
        elif operation == "argmax":
            return aggregator.get_output_tensor()[0][0].cpu()
        else:
            raise NotImplementedError(
                f"Only support operation [per_class, argmax], but got {operation}"
            )

    def predict(self, image: torch.Tensor, operation="per_class"):
        input_tio_image = tio.Image(tensor=image.unsqueeze(0), type=tio.INTENSITY)
        subject = tio.Subject(image=input_tio_image)
        return self._predict(subject, operation)
