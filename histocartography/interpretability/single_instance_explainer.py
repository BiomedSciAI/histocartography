import math
import time
import numpy as np
import torch
import torch.nn as nn

from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN


class SingleInstanceExplainer:
    def __init__(
            self,
            model,
            train_params,
            model_params,
            cuda=False
    ):
        self.model = model
        self.train_params = train_params
        self.model_params = model_params
        self.cuda = cuda

    # Main method
    def explain(
            self, graph, label, unconstrained=False, model="exp"
    ):
        """
        Explain a graph instance
        """

        sub_adj = graph.adjacency_matrix().unsqueeze(dim=0)
        sub_feat = graph.ndata[GNN_NODE_FEAT_IN].unsqueeze(dim=0)
        sub_label = label

        adj = torch.tensor(sub_adj, dtype=torch.float)
        x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)
        pred_label = np.argmax(self.model(graph, graph.ndata[GNN_NODE_FEAT_IN]), axis=0)

        print('Pred label {} | Groud truth label {}'.format(pred_label, label))

        explainer = ExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label,
            args=self.args,
        )
        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()
        explainer.train()
        begin_time = time.time()
        for epoch in range(self.args.num_epochs):
            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            ypred, adj_atts = explainer(unconstrained=unconstrained)
            loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
            loss.backward()

            explainer.optimizer.step()
            if explainer.scheduler is not None:
                explainer.scheduler.step()

            if model != "exp":
                break

            print("finished training in ", time.time() - begin_time, 'density:', explainer.mask_density().item()
                  )
            if model == "exp":
                masked_adj = (
                        explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
                )
            else:
                adj_atts = nn.functional.sigmoid(adj_atts).squeeze()
                masked_adj = adj_atts.cpu().detach().numpy() * sub_adj.squeeze()

        return masked_adj
