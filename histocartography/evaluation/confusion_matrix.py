import torch
from sklearn.metrics.classification import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import io
from copy import deepcopy
from PIL import Image


class ConfusionMatrix:

    def __init__(self, return_img=False):
        """
        Confusion matrix constructor
        """

        self.return_img = return_img
        self.formatting = '.3g'

    def __call__(self, labels, logits):
        """
        Compute the confusion matrix using the labels and logits
        :param labels: (list of torch.LongTensor)
        :param logits: (list of torch.FloatTensor)
        :return: conf_matrix (ndarray)
        """

        # get predictions
        labels = torch.cat(labels, dim=0)
        logits = torch.cat(logits, dim=0)
        predictions = torch.argmax(logits, dim=1)

        # get confusion matrix
        conf_matrix = confusion_matrix(
            y_true=labels.cpu().numpy(),
            y_pred=predictions.cpu().numpy()
        )

        if self.return_img:

            # generate heat map
            scale = max(1., len(conf_matrix) / 10.)
            sns.set(font_scale=0.5)
            plt.figure(figsize=[scale * figax for figax in [6.4, 4.8]])  # default: [6.4, 4.8]
            ax = sns.heatmap(conf_matrix, annot=True, fmt=self.formatting)
            ax.autoscale(tight=True)

            # convert to PIL Image object
            buf = io.BytesIO()
            plt.savefig(buf)
            buf.seek(0)
            pil_img = deepcopy(Image.open(buf))
            pil_img = Image.eval(pil_img, (lambda x: x))
            buf.close()
            return pil_img

        return conf_matrix
