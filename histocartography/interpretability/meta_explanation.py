from histocartography.utils.io import save_image, write_json
from histocartography.interpretability.constants import EXPLANATION_TYPE_SAVE_SUBDIR
from histocartography.evaluation.evaluator import WeightedF1, CrossEntropyLoss, ClusteringQuality
from histocartography.evaluation.classification_report import ClassificationReport
from histocartography.evaluation.nuclei_evaluator import NucleiEvaluator 
from histocartography.utils.visualization import tSNE
from histocartography.utils.draw_utils import plot_tSNE
from histocartography.dataloader.constants import get_number_of_classes

import os
import numpy as np
import torch 

BASE_OUTPUT_PATH = '/dataT/pus/histocartography/Data/explainability/output/'

# The metrics to run on the whole test set are: 
#   - Cross entropy loss of the original prediction  V
#   - Cross entropy loss of the masked prediction    
#   - Weighted F1-score of the original prediction   V
#   - Weigthed F1-score of the masked prediction
#   - Classification report (as a dict) of the original prediction  V
#   - CLassification report (as a dict) of the masked prediction 
#   - tSNE embeddings (as an image) of the original prediction  V
#   - tSNE embeddings (as an image) of the masked prediction
#   - clustering quality of the original predictions    V
#   - clustering quality of the masked predictions 
#   - a set of nuclei metrics...??

# The elements that we need are:
#   - the logits of the original prediction
#   - the logits of the masked prediction (@10% @20%, @50%, etc...)
#   - the latent embeddings of the orginal predictions
#   - the latent embeddings of the masked predictions 
#   - the nuclei labels 

# Extennal elements that we need are:
#   - prior knowledge graph showing the class inter-dependency 
#   - all the priors encoding the correspondence between nuclei importance, nuclei labels for a given class 

VAR_TO_METRIC_FN = {
    '_f1_score': WeightedF1,
    '_ce_loss': CrossEntropyLoss,
    '_classification_report': ClassificationReport,
    # '_clustering': ClusteringQuality
}


CONVERT_SERIALIZABLE_TYPE = {
    torch.Tensor: lambda x: x.item(),
    dict: lambda x: x
}


class MetaBaseExplanation:
    def __init__(
            self,
            config,
            explanations
    ):
        """
        MetaBaseExplanation constructor: Object that is defining an explanation for a given test dataset

        :param config: (dict) configuration parameters 
        :param explanations: (list) all the explanation objects stored in a list 
        """
        self.config = config
        self.explanations = explanations

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):
        raise NotImplementedError('Implementation in subclasses')

    def evaluate(self):
        raise NotImplementedError('Implementation in subclasses')


class MetaGraphExplanation(MetaBaseExplanation):
    def __init__(
            self,
            config,
            explanations
    ):
        """
        Meta Graph Explanation constructor: Object that is defining an explanation for a given test dataset

        :param config: (dict) configuration parameters 
        :param explanations: (list) all the explanation objects stored in a list 

        """

        super(MetaGraphExplanation, self).__init__(config, explanations)

        self.num_classes = get_number_of_classes(config['model_params']['class_split'])
        self.graph_type = config['model_params']['model_type'].replace('_model', '')

        self.save_path = os.path.join(
            BASE_OUTPUT_PATH,
            'gnn',
            str(self.num_classes) + '_class_scenario',
            self.graph_type,
            EXPLANATION_TYPE_SAVE_SUBDIR[config['explanation_type']]
        )

        os.makedirs(self.save_path, exist_ok=True)

        # extract meta information stored in the explanations 
        self._extract_data_from_explanations('logits')
        self._extract_data_from_explanations('latent')
        self._extract_labels()

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):

        # 1. run metrics to derive meta explanation results 
        res = self.evaluate()

        # 3. write json 
        self._encapsulate_meta_explanation(res)
        write_json(os.path.join(self.save_path, 'meta_explanation.json'), self.meta_explanation_as_dict)

    def _encapsulate_meta_explanation(self, res):
        self.meta_explanation_as_dict = {}

        # a. config file
        self.meta_explanation_as_dict['config'] = self.config

        # b. output 
        self.meta_explanation_as_dict['output'] = res

    def evaluate(self):
        """
        Evaluate the quality of the explanation 

        return:
            - res: (dict) (surrogate) metrics so
        """
        res = {}
        for prediction_type in self.explanations[0].explanation_graphs.keys():
            attr_name = 'keep_' + str(int(prediction_type * 100))
            res[attr_name] = {}
            for metric_name, metric_fn in VAR_TO_METRIC_FN.items():
                out = metric_fn()(getattr(self, attr_name + '_logits'), self.labels)  # @TODO: currently all the evaluators are based on logits // will be changed in the future 
                out = CONVERT_SERIALIZABLE_TYPE[type(out)](out)
                res[attr_name][metric_name] = out
        return res

    def _extract_data_from_explanations(self, key, verbose=True):
        for prediction_type in self.explanations[0].explanation_graphs.keys():
            attr_name = 'keep_' + str(int(prediction_type * 100)) + '_' + key
            data = []
            for explanation in self.explanations:
                data.append(explanation.explanation_graphs[1][key])
            data = torch.FloatTensor(data)
            setattr(self, attr_name, data)
            if verbose:
                print('Set new attribute: {} with shape: {}'.format(attr_name, data.shape))

    def _extract_labels(self):
        self.labels = []
        for explanation in self.explanations:
            self.labels.append(explanation.label)
        self.labels = torch.LongTensor(self.labels)

class MetaImageExplanation(MetaBaseExplanation):
    def __init__(
            self,
            config,
            explanations
    ):
        """
        Meta Image Explanation constructor: Object that is defining a meta explanation for a given test datset 

        :param config: (dict) configuration parameters 
        :param explanations: (list) all the explanation objects stored in a list 
        
        """

        super(MetaImageExplanation, self).__init__(config, explanations)

        self.heatmap = heatmap
        self.save_path = '/dataT/pus/histocartography/Data/explainability/output/cnn'

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):
        explanation_as_image = self.draw()
        save_image(os.path.join(self.save_path, self.image_name + '_explanation.png'), explanation_as_image)



