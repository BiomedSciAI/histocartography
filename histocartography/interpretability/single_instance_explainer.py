import time
import numpy as np
import torch

from ..ml.layers.constants import GNN_NODE_FEAT_IN
from .explainer_model import ExplainerModel


class SingleInstanceExplainer:
    def __init__(
            self,
            model,
            train_params,
            model_params,
            cuda=False,
            verbose=False
    ):
        self.model = model
        self.train_params = train_params
        self.model_params = model_params
        self.cuda = cuda
        self.verbose = verbose

    def explain(self, data, label):
        """
        Explain a graph instance
        """

        graph = data[0]

        sub_adj = graph.adjacency_matrix().to_dense().unsqueeze(dim=0)
        sub_feat = graph.ndata[GNN_NODE_FEAT_IN].unsqueeze(dim=0)
        sub_label = label

        adj = torch.tensor(sub_adj, dtype=torch.float)
        x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)
        pred_label = np.argmax(self.model(data).cpu().detach().numpy().squeeze(), axis=0)

        print('Pred label {} | Groud truth label {}'.format(
            pred_label,
            label)
        )

        explainer = ExplainerModel(
            model=self.model,
            adj=adj,
            x=x,
            label=label,
            model_params=self.model_params,
            train_params=self.train_params
        )

        if self.cuda:
            explainer = explainer.cuda()

        self.model.eval()
        explainer.train()

        begin_time = time.time()
        for epoch in range(self.train_params['num_epochs']):
            ypred = explainer()
            loss = explainer.loss(ypred)

            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            loss.backward()
            explainer.optimizer.step()

            # if explainer.scheduler is not None:
            #     explainer.scheduler.step()

        print("Training time: {} with density {}".format(
            time.time() - begin_time,
            explainer.mask_density().item())
        )

        masked_adj = (
                explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.cpu().detach().numpy().squeeze()
        )

        masked_feats = explainer.x * explainer.feat_mask

        return masked_adj, masked_feats
