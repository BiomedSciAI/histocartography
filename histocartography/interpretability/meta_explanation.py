from histocartography.utils.io import save_image, write_json
from histocartography.interpretability.constants import EXPLANATION_TYPE_SAVE_SUBDIR
from histocartography.evaluation.evaluator import WeightedF1, CrossEntropyLoss, ClusteringQuality, ExpectedClassShiftWithLogits, WeightedExpectedClassShiftWithLogits
from histocartography.evaluation.classification_report import ClassificationReport
from histocartography.evaluation.nuclei_evaluator import NucleiEvaluator 
from histocartography.utils.visualization import tSNE
from histocartography.utils.draw_utils import plot_tSNE
from histocartography.dataloader.constants import get_number_of_classes, get_label_to_tumor_type
from histocartography.interpretability.constants import FIVE_CLASS_DEPENDENCY_GRAPH, SEVEN_CLASS_DEPENDENCY_GRAPH, THREE_CLASS_DEPENDENCY_GRAPH

import os
import numpy as np
import torch 

BASE_OUTPUT_PATH = '/dataT/pus/histocartography/Data/explainability/output/'


VAR_TO_METRIC_FN = {
    '_f1_score': ({}, WeightedF1, '_logits'),
    '_ce_loss': ({}, CrossEntropyLoss, '_logits'),
    '_classification_report': ({}, ClassificationReport, '_logits'),
    '_expected_class_shift': ({'knowledge_graph': FIVE_CLASS_DEPENDENCY_GRAPH}, ExpectedClassShiftWithLogits, '_logits'),
    '_weighted_expected_class_shift': ({'knowledge_graph': FIVE_CLASS_DEPENDENCY_GRAPH}, WeightedExpectedClassShiftWithLogits, '_logits'),  # HACK ALERT: set to 3-class scenario
    'metric1': ({'knowledge_graph': WHATEVER_WE_NEED}, Metric1, '_nuclei_label'),    # @TODO: KG + nuclei labels + TRoI-level prediction 
    'metric2': ({'knowledge_graph': WHATERVER_WE_NEED}, Metric2, '_nuclei_label')    # @TODO: KG + nuclei labels + TRoI-level prediction 
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
        self.num_classes = get_number_of_classes(config['model_params']['class_split'])
        self._extract_labels()

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):
        raise NotImplementedError('Implementation in subclasses')

    def evaluate(self):
        raise NotImplementedError('Implementation in subclasses')

    def _extract_labels(self):
        self.labels = []
        for explanation in self.explanations:
            self.labels.append(explanation.label)
        self.labels = torch.LongTensor(self.labels)

    def _encapsulate_meta_explanation(self, res):
        self.meta_explanation_as_dict = {}
        # a. config file
        self.meta_explanation_as_dict['config'] = self.config
        # b. output 
        self.meta_explanation_as_dict['output'] = res


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
        self._extract_data_from_explanations('nuclei_label')

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):

        # 1. run metrics to derive meta explanation results 
        res = self.evaluate()

        # 2. write json 
        self._encapsulate_meta_explanation(res)
        write_json(os.path.join(self.save_path, 'meta_explanation.json'), self.meta_explanation_as_dict)

        # 3. draw tSNE projection
        self._run_and_draw_tsne()

    def _run_and_draw_tsne(self):
        eval_tsne = tSNE()
        for prediction_type in self.explanations[0].explanation_graphs.keys():
            attr_name = 'keep_' + str(int(prediction_type * 100))
            low_dim_emb = eval_tsne(getattr(self, attr_name + '_latent'))
            plot_tSNE(low_dim_emb, self.labels, os.path.join(self.save_path, attr_name + '_tsne.png'), get_label_to_tumor_type(self.config['model_params']['class_split']))

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
            for metric_name, (metric_arg, metric_fn, metric_var) in VAR_TO_METRIC_FN.items():
                out = metric_fn(**metric_arg)(getattr(self, attr_name + metric_var), self.labels)
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

        self.save_path = os.path.join(
            BASE_OUTPUT_PATH,
            'cnn',
            str(self.num_classes) + '_class_scenario',
            EXPLANATION_TYPE_SAVE_SUBDIR[config['explanation_type']]
        )

        os.makedirs(self.save_path, exist_ok=True)

        self._extract_data_from_explanations('logits')

    def read(self):
        raise NotImplementedError('Implementation in subclasses')

    def write(self):
        
        # 1. run metrics to derive meta explanation results 
        res = self.evaluate()

        # 2. write json 
        self._encapsulate_meta_explanation(res)
        write_json(os.path.join(self.save_path, 'meta_explanation.json'), self.meta_explanation_as_dict)

    def evaluate(self):
        res = {}
        for metric_name, (metric_arg, metric_fn) in VAR_TO_METRIC_FN.items():
            out = metric_fn(**metric_arg)(getattr(self, 'all_logits'), self.labels)
            out = CONVERT_SERIALIZABLE_TYPE[type(out)](out)
            res['all' + metric_name] = out
            print('Name:', 'all' + metric_name, out)
        return res

    def _extract_data_from_explanations(self, key, verbose=True):
        attr_name = 'all_' + key
        data = []
        for explanation in self.explanations:
            data.append(getattr(explanation, key))
        data = torch.FloatTensor(data)
        setattr(self, attr_name, data)
        if verbose:
            print('Set new attribute: {} with shape: {}'.format(attr_name, data.shape))




