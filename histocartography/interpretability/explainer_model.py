import math
import torch
import torch.nn as nn
import torch.optim as optim


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

        # set data
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
        self.mask, self.mask_bias = self._build_edge_mask(num_nodes, init_strategy=init_strategy)
        self.feat_mask = self._build_feat_mask(x.size(-1), init_strategy="constant")
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)

        # group them
        params = [self.mask, self.feat_mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)

        if self.cuda:
            self.diag_mask = self.diag_mask.cuda()

        # build optimizer
        self._build_optimizer(params, train_params)

        # build loss reg weights
        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0
        }

    def _build_optimizer(self, params, train_params):
        self.optimizer = optim.Adam(params, lr=train_params['lr'], weight_decay=train_params['weight_decay'])
        # self.scheduler = optim.lr_scheduler.StepLR(
        #     self.optimizer,
        #     step_size=train_params['opt_decay_step'],
        #     gamma=train_params['opt_decay_rate']
        # )

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
        return masked_adj * self.diag_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def forward(self):

        self.masked_adj = self._masked_adj()

        feat_mask = (
            torch.sigmoid(self.feat_mask)
            if self.use_sigmoid
            else self.feat_mask
        )
        x = self.x * feat_mask

        # build a graph from the new x & adjacency matrix...
        graph = [self.masked_adj, x]
        ypred = self.model(graph)
        res = nn.Softmax(dim=0)(ypred)

        return res

    def loss(self, pred):
        """
        Args:
            pred: prediction made by current model
        """

        gt_label_node = self.label
        logit = pred[gt_label_node]
        pred_loss = -torch.log(logit)  # @TODO: change if multi class classification ?

        # size
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        else:
            raise ValueError('Unsupported mask activation {}. Options'
                             'are "sigmoid", "ReLU"'.format(self.mask_act))
        size_loss = self.coeffs["size"] * torch.sum(mask)

        feat_mask = (
            torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss + feat_size_loss

        return loss