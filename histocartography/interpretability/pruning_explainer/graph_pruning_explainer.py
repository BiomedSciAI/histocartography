import torch
from tqdm import tqdm
from copy import deepcopy
import dgl 

from ...ml.layers.constants import GNN_NODE_FEAT_IN
from .explainer_model import ExplainerModel
from ..base_explainer import BaseExplainer
from ...utils.torch import torch_to_numpy
from ...utils.graph import set_graph_on_cpu


class GraphPruningExplainer(BaseExplainer):
    def __init__(
        self,
        entropy_loss_weight: float = 1.0,
        size_loss_weight: float = 0.05,
        ce_loss_weight: float = 10.0,
        node_thresh: float = 0.05,
        mask_init_strategy: str = "normal",
        mask_activation: str = "sigmoid",
        num_epochs: int = 500,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        **kwargs
    ) -> None:
        """
        Graph Pruning Explainer (GNNExplainer) constructor 

        Args:
            entropy_loss_weight (float): how much weight to put on the
                                         element-wise entropy loss term.
                                         Default to  1.0.
            size_loss_weight (float): how much weight to put on the mask
                                      size. Default to 0.05.
            ce_loss_weight float): how much weight to put on the cross-
                                   entropy loss term. Default to 10.0.
            node_thresh (float): Threshold value to set deactivate node.
                                 Default to 0.05.
            mask_init_strategy (str): Initialization strategy for the
                                      mask. Default to "normal" (ie all 1's).
            mask_activation (str): Mask activation function. Default to "sigmoid".
            num_epochs (int): Number of epochs used for training the mask. 
                              Default to 500.
            lr (float): Learning rate. Default to 0.01.
            weight_decay (float): Weight decay. Default to 5e-4. 
        """
    
        super(GraphPruningExplainer, self).__init__(**kwargs)

        self.node_thresh = node_thresh
        self.train_params = {
            'num_epochs': num_epochs,
            'lr': lr,
            'weight_decay': weight_decay
        }
        self.model_params = {
            'loss': {
                'node_ent': entropy_loss_weight,
                'node': size_loss_weight,
                'ce': ce_loss_weight 
            },
            'node_thresh': node_thresh ,
            'init': mask_init_strategy,
            'mask_activation': mask_activation 
        }

        self.node_feats_explanation = None
        self.probs_explanation = None
        self.node_importance = None

    def process(self, graph: dgl.DGLGraph, label: int = None):
        """
        Explain a graph instance
        
        Args:
            graph (dgl.DGLGraph): Input graph to explain
            label (int): Label attached to the graph. Required. 
        """

        sub_adj = graph.adjacency_matrix().to_dense().unsqueeze(dim=0)
        sub_feat = graph.ndata[GNN_NODE_FEAT_IN].unsqueeze(dim=0)

        adj = torch.tensor(sub_adj, dtype=torch.float).to(self.device)
        x = torch.tensor(sub_feat, dtype=torch.float).to(self.device)

        init_logits = self.model([graph])
        init_logits = init_logits.cpu().detach()
        init_probs = torch.nn.Softmax()(init_logits)
        init_pred_label = torch.argmax(init_logits, dim=1).squeeze()

        explainer = ExplainerModel(
            model=deepcopy(self.model),
            adj=adj,
            x=x,
            init_probs=init_probs.to(self.device),
            model_params=self.model_params,
            train_params=self.train_params,
            cuda=self.cuda
        ).to(self.device)

        self.node_feats_explanation = x
        self.probs_explanation = init_probs
        self.node_importance = torch_to_numpy(explainer._get_node_feats_mask())

        self.model.eval()
        explainer.train()

        # Init training stats
        init_probs = init_probs.numpy().squeeze()
        init_num_nodes = adj.shape[-1]
        loss = torch.FloatTensor([10000.])

        # log description
        desc = self._set_pbar_desc()
        pbar = tqdm(range(self.train_params['num_epochs']), desc=desc, unit='step')
    
        for step in pbar:
            logits, masked_feats = explainer()
            loss = explainer.loss(logits)

            # Compute number of non zero elements in the masked adjacency
            node_importance = explainer._get_node_feats_mask()
            node_importance[node_importance < self.node_thresh] = 0.
            masked_feats = masked_feats * torch.stack(masked_feats.shape[-1] * [node_importance], dim=1).unsqueeze(dim=0).to(torch.float)
            probs = torch.nn.Softmax()(logits.cpu().squeeze()).detach().numpy()
            pred_label = torch.argmax(logits, dim=0).squeeze()

            # handle early stopping if the labels is changed
            if pred_label.item() == init_pred_label:
                self.node_feats_explanation = masked_feats
                self.probs_explanation = probs
                self.node_importance = torch_to_numpy(node_importance)
            else:
                print('Predicted label changed. Early stopping.')
                break

            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            explainer.optimizer.step()

        node_importance = self.node_importance
        logits = init_logits.cpu().detach().numpy()

        return node_importance, logits 

    def _set_pbar_desc(self):
        desc = "Process:"
        return desc




















