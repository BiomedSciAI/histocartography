from tqdm import tqdm
from copy import deepcopy
import dgl
import math
from scipy.stats import entropy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import importlib

from ..ml.layers.constants import GNN_NODE_FEAT_IN
from .base_explainer import BaseExplainer
from ..utils.torch import torch_to_numpy
from ..utils import is_box_url, download_box_link


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_MODULE = 'histocartography.ml'


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

        # GNNExplainer needs to work with dense layers, and not with DGL
        # objects.
        self.model = self._convert_to_dense_gnn_model()

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
            'node_thresh': node_thresh,
            'init': mask_init_strategy,
            'mask_activation': mask_activation
        }

        self.node_feats_explanation = None
        self.probs_explanation = None
        self.node_importance = None

    def _convert_to_dense_gnn_model(self):

        # load DGL-based model
        dgl_model = self.model

        # rebuild model by replacing DGL layers by Dense layers.
        model_name = dgl_model.__class__.__name__
        dgl_gnn_params = dgl_model.gnn_params
        dgl_layer_type = dgl_gnn_params['layer_type']
        assert dgl_layer_type == 'gin_layer', "Only GIN layers are supported for using GNNExplainer."
        dense_gnn_params = deepcopy(dgl_gnn_params)
        dense_gnn_params['layer_type'] = 'dense_' + dgl_layer_type

        module = importlib.import_module(MODEL_MODULE)
        model = getattr(module, model_name)(
            dense_gnn_params,
            dgl_model.classification_params,
            dgl_model.node_dim,
            num_classes=dgl_model.num_classes
        )

        # copy weights from DGL layers to dense layers.
        def is_int(s):
            try:
                int(s)
                return True
            except BaseException:
                return False

        for n, p in dgl_model.named_parameters():
            split = n.split('.')
            to_eval = 'model'
            for s in split:
                if is_int(s):
                    to_eval += '[' + s + ']'
                else:
                    to_eval += '.'
                    to_eval += s
            exec(to_eval + '=' + 'p')

        return model

    def _process(self, graph: dgl.DGLGraph, label: int = None):
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

        init_logits = self.model(graph)
        init_logits = init_logits.cpu().detach()
        init_probs = torch.nn.Softmax()(init_logits)
        init_pred_label = torch.argmax(init_logits, dim=1).squeeze()

        explainer = ExplainerModel(
            model=deepcopy(self.model),
            adj=adj,
            x=x,
            init_probs=init_probs.to(self.device),
            model_params=self.model_params,
            train_params=self.train_params
        ).to(self.device)

        self.node_feats_explanation = x
        self.probs_explanation = init_probs
        self.node_importance = torch_to_numpy(explainer._get_node_feats_mask())

        self.model.eval()
        explainer.train()

        # Init training stats
        init_probs = init_probs.numpy().squeeze()
        loss = torch.FloatTensor([10000.])

        # log description
        desc = self._set_pbar_desc()
        pbar = tqdm(
            range(
                self.train_params['num_epochs']),
            desc=desc,
            unit='step')

        for _ in pbar:
            logits, masked_feats = explainer()
            loss = explainer.loss(logits)

            # Compute number of non zero elements in the masked adjacency
            node_importance = explainer._get_node_feats_mask()
            node_importance[node_importance < self.node_thresh] = 0.
            masked_feats = masked_feats * \
                torch.stack(masked_feats.shape[-1] * [node_importance], dim=1).unsqueeze(dim=0).to(torch.float)
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


class ExplainerModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        adj: torch.tensor,
        x: torch.tensor,
        init_probs: torch.tensor,
        model_params: dict,
        train_params: dict,
        use_sigmoid: bool = True,
    ):
        """
        Explainer constructor.

        Args:
            model (nn.Module): Torch model.
            adj (torch.tensor): Adjacency matrix.
            x (torch.tensor): Node features.
            init_probs (torch.tensor:): Prediction on the whole graph.
            model_params (dict): Model params for learning mask.
            train_params (dict): Training params for learning mask.
            use_sigmoid (bool): Default to True.
        """

        super(ExplainerModel, self).__init__()

        # set data & model
        self.device = DEVICE
        self.adj = adj
        self.x = x
        self.model = model.to(self.device)
        self.init_probs = init_probs
        self.label = torch.argmax(init_probs, dim=1)

        # set model parameters
        self.mask_act = model_params['mask_activation']
        init_strategy = model_params['init']
        self.mask_bias = None
        self.use_sigmoid = use_sigmoid

        # build learnable parameters: edge mask & feat mask (& node_mask)
        num_nodes = adj.size()[1]
        self.num_nodes = num_nodes
        self.mask, _ = self._build_edge_mask(
            num_nodes, init_strategy=init_strategy)
        self.node_mask = self._build_node_mask(
            num_nodes, init_strategy='const')
        self.diag_mask = torch.ones(
            num_nodes, num_nodes) - torch.eye(num_nodes)
        self.diag_mask = self.diag_mask.to(self.device)

        # group them
        params = [self.mask, self.node_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)

        # build optimizer
        self._build_optimizer(params, train_params)

        self.coeffs = model_params['loss']

    def _build_optimizer(self, params, train_params):
        self.optimizer = optim.Adam(
            params,
            lr=train_params['lr'],
            weight_decay=train_params['weight_decay'])

    def _build_edge_mask(
            self,
            num_nodes,
            init_strategy="normal",
            const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.mask_bias is not None:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def _build_node_mask(
            self,
            num_nodes,
            init_strategy="normal",
            const_val=1.0):
        node_mask = nn.Parameter(torch.FloatTensor(num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                node_mask.normal_(1.0, std)
        elif init_strategy == "const":
            nn.init.constant_(node_mask, const_val)
        return node_mask

    def _get_adj_mask(self, with_zeroing=False):
        if self.mask_act == "sigmoid":
            sym_mask = self.sigmoid(self.mask, t=2)
        elif self.mask_act == "relu":
            sym_mask = nn.ReLU()(self.mask)
        else:
            raise ValueError('Unsupported mask activation {}. Options'
                             'are "sigmoid", "ReLU"'.format(self.mask_act))
        sym_mask = (sym_mask + sym_mask.t()) / 2
        if with_zeroing:
            sym_mask = (
                (self.adj != 0).to(
                    self.device).to(
                    torch.float) *
                sym_mask)
        return sym_mask

    def _masked_adj(self):
        sym_mask = self._get_adj_mask()
        adj = adj.to(self.device)
        masked_adj = adj * sym_mask
        if self.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        masked_adj = masked_adj * self.diag_mask
        return masked_adj

    def _get_node_feats_mask(self):
        if self.mask_act == "sigmoid":
            node_mask = self.sigmoid(self.node_mask, t=10)
        elif self.mask_act == "relu":
            node_mask = nn.ReLU()(self.node_mask)
        else:
            raise ValueError('Unsupported mask activation {}. Options'
                             'are "sigmoid", "ReLU"'.format(self.mask_act))
        return node_mask

    @staticmethod
    def sigmoid(x, t=1):
        return 1 / (1 + torch.exp(-t * x))

    def _masked_node_feats(self):
        node_mask = self._get_node_feats_mask()
        x = self.x * \
            torch.stack(self.x.shape[-1] * [node_mask], dim=1).unsqueeze(dim=0)
        return x

    def forward(self):
        """
        Forward pass.
        """
        masked_x = self._masked_node_feats()
        graph = [self.adj, masked_x]
        ypred = self.model(graph)
        return ypred, masked_x

    def distillation_loss(self, inner_logits):
        """
        Compute distillation loss.
        """
        log_output = nn.LogSoftmax(dim=1)(inner_logits)
        cross_entropy = self.init_probs * log_output
        return -torch.mean(torch.sum(cross_entropy, dim=1))

    def loss(self, pred: torch.tensor):
        """
        Compute new overall loss given current prediction.
        Args:
            pred (torch.tensor): Prediction made by current model.
        """

        # 1. cross-entropy + distillation loss
        ce_loss = F.cross_entropy(pred.unsqueeze(dim=0), self.label)
        distillation_loss = self.distillation_loss(pred.unsqueeze(dim=0))
        alpha = torch.FloatTensor([entropy(torch.nn.Softmax()(pred).cpu().detach(
        ).numpy())]) / torch.log(torch.FloatTensor([self.init_probs.shape[1]]))
        alpha = alpha.to(self.device)
        pred_loss = self.coeffs['ce'] * \
            (alpha * ce_loss + (1 - alpha) * distillation_loss)

        # 2. node loss
        node_mask = self._get_node_feats_mask()
        node_loss = self.coeffs["node"] * torch.sum(node_mask)

        # 3. node entropy loss
        node_ent = -node_mask * \
            torch.log(node_mask) - (1 - node_mask) * torch.log(1 - node_mask)
        node_ent_loss = self.coeffs["node_ent"] * torch.mean(node_ent)

        # 4. sum all the losses
        loss = pred_loss + node_loss + node_ent_loss

        return loss
