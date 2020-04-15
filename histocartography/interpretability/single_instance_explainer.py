import time
import numpy as np
import torch
from tqdm import tqdm

from ..ml.layers.constants import GNN_NODE_FEAT_IN
from ..dataloader.constants import LABEL_TO_TUMOR_TYPE
from .explainer_model import ExplainerModel
from histocartography.utils.io import get_device


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
        self.device = get_device(self.cuda)
        self.verbose = verbose
        self.adj_thresh = model_params['adj_thresh']
        self.node_thresh = model_params['node_thresh']

        self.adj_explanation = None
        self.node_feats_explanation = None
        self.logit_explanation = None

    def explain(self, data, label):
        """
        Explain a graph instance
        """

        graph = data[0]

        sub_adj = graph.adjacency_matrix().to_dense().unsqueeze(dim=0)
        sub_feat = graph.ndata[GNN_NODE_FEAT_IN].unsqueeze(dim=0)
        sub_label = label

        adj = torch.tensor(sub_adj, dtype=torch.float).to(self.device)
        x = torch.tensor(sub_feat, dtype=torch.float).to(self.device)
        label = torch.tensor(sub_label, dtype=torch.long).to(self.device)
        init_logits = self.model(data).cpu().detach()
        init_probs = torch.nn.Softmax()(init_logits).numpy().squeeze()
        init_pred_label = torch.argmax(init_logits, axis=1).squeeze()
        self.adj_explanation = adj
        self.node_feats_explanation = x
        self.logit_explanation = init_logits

        explainer = ExplainerModel(
            model=self.model,
            adj=adj,
            x=x,
            label=label,
            model_params=self.model_params,
            train_params=self.train_params,
            cuda=self.cuda
        ).to(self.device)

        self.model.eval()
        explainer.train()

        # Init training stats
        init_non_zero_elements = (adj != 0).sum()
        init_num_nodes = adj.shape[-1]
        density = 1.0
        loss = torch.FloatTensor([10000.])
        desc = "Nodes {} / {} | Edges {} / {} | Density {} | Loss {} | Label {}" \
               "| N {} / {} | B {} / {} | ATY {} / {} | DCIS {} / {} | I {} / {}".format(
            init_num_nodes, init_num_nodes,
            init_non_zero_elements, init_non_zero_elements,
            density,
            loss.item(), 
            LABEL_TO_TUMOR_TYPE[str(label.item())],
            round(float(init_probs[0]), 2), round(float(init_probs[0]), 2),
            round(float(init_probs[1]), 2), round(float(init_probs[1]), 2),
            round(float(init_probs[3]), 2), round(float(init_probs[3]), 2),
            round(float(init_probs[4]), 2), round(float(init_probs[4]), 2),
            round(float(init_probs[2]), 2), round(float(init_probs[2]), 2)
        )
        pbar = tqdm(range(self.train_params['num_epochs']), desc=desc, unit='step')
    
        for step in pbar:
            logits, masked_adj, masked_feats = explainer()
            loss = explainer.loss(logits)

            # Compute number of non zero elements in the masked adjacency
            masked_adj = (masked_adj > self.adj_thresh).to(self.device).to(torch.float) * masked_adj
            masked_feats = (masked_feats > self.node_thresh).to(self.device).to(torch.float) * masked_feats
            probs = torch.nn.Softmax()(logits.cpu().squeeze()).detach().numpy()
            non_zero_elements = (masked_adj != 0).sum()
            density = round(non_zero_elements.item() / init_non_zero_elements.item(), 2)
            num_nodes = torch.sum(masked_feats.sum(dim=-1) != 0.)
            pred_label = torch.argmax(logits, axis=0).squeeze()

            desc = "Nodes {} / {} | Edges {} / {} | Density {} | Loss {} | Label {}" \
                   "| N {} / {} | B {} / {} | ATY {} / {} | DCIS {} / {} | I {} / {}".format(
                num_nodes, init_num_nodes,
                non_zero_elements, init_non_zero_elements,
                density,
                round(loss.item(), 2),
                LABEL_TO_TUMOR_TYPE[str(label.item())],
                round(float(probs[0]), 2), round(float(init_probs[0]), 2),
                round(float(probs[1]), 2), round(float(init_probs[1]), 2),
                round(float(probs[3]), 2), round(float(init_probs[3]), 2),
                round(float(probs[4]), 2), round(float(init_probs[4]), 2),
                round(float(probs[2]), 2), round(float(init_probs[2]), 2)
            )

            pbar.set_description(desc)

            # handle early stopping if the labels is changed
            if pred_label.item() == init_pred_label:
                self.adj_explanation = masked_adj
                self.node_feats_explanation = masked_feats
                self.logit_explanation = logits
            else:
                print('Predicted label changed. Early stopping.')
                break

            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            loss.backward()
            explainer.optimizer.step()

        # forward pass
        probs = torch.nn.Softmax()(self.logit_explanation.cpu().squeeze()).detach().numpy()
        masked_adj = (self.adj_explanation > self.adj_thresh).to(self.device).to(torch.float) * self.adj_explanation
        masked_feats = (self.node_feats_explanation > self.node_thresh).to(self.device).to(torch.float) * self.node_feats_explanation

        return masked_adj.squeeze(), masked_feats.squeeze(), init_probs, probs
