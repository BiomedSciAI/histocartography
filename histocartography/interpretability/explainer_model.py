import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 


class ExplainerModel(nn.Module):
    def __init__(
        self,
        model,
        adj,
        x,
        label,
        model_params,
        train_params,
        cuda=False,
        use_sigmoid=True,
    ):

        super(ExplainerModel, self).__init__()

        # set data & model
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label

        # set model parameters
        self.cuda = cuda
        self.mask_act = model_params['mask_activation']
        init_strategy = model_params['init']
        self.mask_bias = None
        self.use_sigmoid = use_sigmoid

        # build learnable parameters: edge mask & feat mask
        num_nodes = adj.size()[1]
        self.num_nodes = num_nodes
        self.mask, _ = self._build_edge_mask(num_nodes, init_strategy=init_strategy)
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)

        # group them
        params = [self.mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)

        if self.cuda:
            self.diag_mask = self.diag_mask.cuda()

        # build optimizer
        self._build_optimizer(params, train_params)

        # build loss reg weights
        self.coeffs = {
            "size": 0.005,
            "ent": 1.0
        }

    def _build_optimizer(self, params, train_params):
        self.optimizer = optim.Adam(params, lr=train_params['lr'], weight_decay=train_params['weight_decay'])

    def _build_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def _build_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
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

    def _masked_adj(self):

        # @TODO: testing thresholding the adj to encourage even further sparsity
        # low_thresh = torch.zeros(self.num_nodes, self.num_nodes)
        # high_thresh = torch.ones(self.num_nodes, self.num_nodes)
        # sym_mask = torch.where(self.mask > 0.5, low_thresh, high_thresh)

        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        else:
            raise ValueError('Unsupported mask activation {}. Options'
                             'are "sigmoid", "ReLU"'.format(self.mask_act))

        sym_mask = (sym_mask + sym_mask.t()) / 2

        adj = self.adj.cuda() if self.cuda else self.adj

        masked_adj = adj * sym_mask

        if self.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2

        masked_adj = masked_adj * self.diag_mask

        return masked_adj

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def forward(self):

        masked_adj = self._masked_adj()
        x = self.x

        # build a graph from the new x & adjacency matrix...
        graph = [masked_adj, x]

        # print number of non zero elements in the adjacency:
        non_zero_elements = (masked_adj != 0).sum()
        print('Number of non-zero elements:', non_zero_elements)

        ypred = self.model(graph)

        return ypred, masked_adj, x

    def loss(self, pred):
        """
        Args:
            pred: prediction made by current model
        """

        # 1. cross-entropy loss
        pred_loss = F.cross_entropy(pred.unsqueeze(dim=0), self.label)

        # 2. size loss
        mask = ((self.adj != 0) * self.mask).squeeze()
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(mask)
        else:
            raise ValueError('Unsupported mask activation {}. Options'
                             'are "sigmoid", "ReLU"'.format(self.mask_act))
        size_loss = self.coeffs["size"] * torch.sum(mask)

        # 3. mask entropy loss
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent[mask_ent != mask_ent] = 0
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        loss = pred_loss + size_loss

        print('Loss: {} | Mask density: {} | Prediction: {}'.format(
            loss.item(),
            self.mask_density().item(),
            self.label == torch.argmax(pred).item()
        ))
        print('Prediction:', pred)

        return loss