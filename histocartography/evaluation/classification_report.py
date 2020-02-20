import torch
from sklearn.metrics.classification import classification_report


class ClassificationReport:

    def __init__(self):
        """
        Classification report constructor
        :param num_classes: (int) number of classes
        """

    def __call__(self, logits, labels):
        """
        Compute classification report using the labels and logits
        :param labels: (list of torch.LongTensor)
        :param logits: (list of torch.FloatTensor)
        :return: report (dict)
        """

        # get predictions
        # labels = torch.cat(labels, dim=0)
        # logits = torch.cat(logits, dim=0)
        predictions = torch.argmax(logits, dim=1)

        # get classification report
        report = classification_report(
            y_true=labels.cpu().numpy(),
            y_pred=predictions.cpu().numpy(),
            output_dict=True
        )

        return report
