from histocartography.utils.io import save_image, write_json
from histocartography.interpretability.constants import EXPLANATION_TYPE_SAVE_SUBDIR
from histocartography.evaluation.evaluator import WeightedF1, CrossEntropyLoss, ClusteringQuality
from histocartography.evaluation.classification_report import ClassificationReport
from histocartography.evaluation.nuclei_evaluator import NucleiEvaluator 
from histocartography.utils.visualization import tSNE
from histocartography.utils.draw_utils import plot_tSNE

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
    '_clustering': ClusteringQuality
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
        self.exlanations = explanations

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):
        raise NotImplementedError('Implementation in subclasses')

    def evaluate(self):
        raise NotImplementedError('Implementation in subclasses')


class MetaGraphExplanation(BaseExplanation):
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

        super(GraphExplanation, self).__init__(config, explanations)

        # define save path 
        self.save_path = os.path.join(
            BASE_OUTPUT_PATH,
            'gnn',
            EXPLANATION_TYPE_SAVE_SUBDIR[config['explanation_type']]
        )

        # extract meta information stored in the explanations 
        self._extract_original_logits()
        self._extract_original_latent_embeddings()

        self._extract_labels()

        self._extract_masked_logits()
        self._extract_masked_latent_embeddings()

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):

        # 1. run metrics to derive meta explanation results 
        res = self.evaluate()

        # 3. write json 
        self._encapsulate_meta_explanation(res)
        write_json(os.path.join(self.save_path, + 'meta_explanation.json'), self.meta_explanation)

    def _encapsulate_meta_explanation(self):
        self.explanation_as_dict = {}

        # a. config file
        self.explanation_as_dict['config'] = self.config

        # b. output 
        self.explanation_as_dict['output'] = res
        # self.explanation_as_dict['output']['label_index'] = self.label.item()

        # # 3-c original graph properties
        # self.explanation_as_dict['output']['original'] = {}
        # self.explanation_as_dict['output']['original']['logits'] = list(np.around(self.original_prediction.cpu().detach().numpy(), 2).astype(float))

        # # 3-d explanation graph properties
        # self.explanation_as_dict['output']['explanation'] = {}
        # self.explanation_as_dict['output']['explanation']['number_of_nodes'] = self.explanation_graph.number_of_nodes()
        # self.explanation_as_dict['output']['explanation']['number_of_edges'] = self.explanation_graph.number_of_edges()
        # if self.explanation_prediction is not None:
        #     self.explanation_as_dict['output']['explanation']['logits'] = list(np.around(self.explanation_prediction.cpu().detach().numpy(), 2).astype(float))

    def evaluate(self):
        """
        Evaluate the quality of the explanation 

        return:
            - res: (dict) (surrogate) metrics so
        """
        res = {}
        for prediction_type in ['original', 'masked']:
            res[prediction_type] = {}
            for metric_name, metric_fn in VAR_TO_METRIC_FN:
                res[prediction_type][metric_name] = metric_fn(getattr(self, prediction_type + '_logits'), self.labels)
        return res

    def _extract_original_logits():
        self.original_logits = []
        for explanation in self.explanations:
            self.original_logits.append(explanation.original_prediction)
        self.original_logits = torch.stack(self.original_logits, dim=0)

    def _extract_labels():
        self.labels = []
        for explanation in self.explanations:
            self.original_logits.append(explanation.label)
        self.labels = torch.LongTensor(self.labels)

    def _extract_masked_logits():
        self.masked_logits = []
        for explanation in self.explanations:
            self.original_logits.append(explanation.explanation_prediction)
        self.masked_logits = torch.stack(self.masked_logits, dim=0)

    def _extract_original_latent_embeddings():
        self.original_latent_embeddings = []
        for explanation in self.explanations:
            self.original_logits.append(explanation.original_latent_embedding)
        self.original_latent_embeddings = torch.stack(self.original_latent_embeddings, dim=0)

    def _extract_masked_latent_embeddings():
        self.masked_latent_embeddings = []
        for explanation in self.explanations:
            self.original_logits.append(explanation.masked_latent_embedding)
        self.masked_latent_embeddings = torch.stack(self.masked_latent_embeddings, dim=0)

class MetaImageExplanation(BaseExplanation):
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

        super(ImageExplanation, self).__init__(config, image, image_name, original_prediction, label)

        self.heatmap = heatmap
        self.save_path = '/dataT/pus/histocartography/Data/explainability/output/cnn'

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):
        explanation_as_image = self.draw()
        save_image(os.path.join(self.save_path, self.image_name + '_explanation.png'), explanation_as_image)



