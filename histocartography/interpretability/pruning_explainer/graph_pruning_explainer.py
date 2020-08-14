import torch
from tqdm import tqdm

from ...ml.layers.constants import GNN_NODE_FEAT_IN
from ...dataloader.constants import get_label_to_tumor_type
from .explainer_model import ExplainerModel
from histocartography.utils.io import get_device
from ..base_explainer import BaseExplainer
from ..explanation import GraphExplanation
from histocartography.utils.graph import adj_to_networkx


class GraphPruningExplainer(BaseExplainer):
    def __init__(
            self,
            model,
            config,
            cuda=False,
            verbose=False
    ):

        super(GraphPruningExplainer, self).__init__(model, config, cuda, verbose)

        self.train_params = self.config['train_params']
        self.model_params = self.config['model_params']
        self.label_to_tumor_type = get_label_to_tumor_type(self.model_params['class_split'])
        self.verbose = verbose
        self.adj_thresh = self.model_params['adj_thresh']
        self.node_thresh = self.model_params['node_thresh']

        self.adj_explanation = None
        self.node_feats_explanation = None
        self.probs_explanation = None
        self.node_importance = None

    def explain(self, data, label):
        """
        Explain a graph instance
        """

        if self.cuda:
            self.model = self.model.cuda()

        graph = data[0]
        image = data[1]
        image_name = data[2]

        sub_adj = graph.adjacency_matrix().to_dense().unsqueeze(dim=0)
        sub_feat = graph.ndata[GNN_NODE_FEAT_IN].unsqueeze(dim=0)
        sub_label = label

        adj = torch.tensor(sub_adj, dtype=torch.float).to(self.device)
        x = torch.tensor(sub_feat, dtype=torch.float).to(self.device)
        label = torch.tensor(sub_label, dtype=torch.long).to(self.device)

        if self.cuda:
            self.model = self.model.cuda()
        init_logits = self.model(data)
        init_logits = init_logits.cpu().detach()
        init_probs = torch.nn.Softmax()(init_logits)
        init_pred_label = torch.argmax(init_logits, dim=1).squeeze()

        explainer = ExplainerModel(
            model=self.model,
            adj=adj,
            x=x,
            init_probs=init_probs.to(self.device),
            model_params=self.model_params,
            train_params=self.train_params,
            cuda=self.cuda
        ).to(self.device)

        self.adj_explanation = adj
        self.node_feats_explanation = x
        self.probs_explanation = init_probs
        self.node_importance = explainer._get_node_feats_mask().cpu().detach().numpy()

        self.model.eval()
        explainer.train()

        # Init training stats
        init_probs = init_probs.numpy().squeeze()
        init_non_zero_elements = (adj != 0).sum()
        init_num_nodes = adj.shape[-1]
        density = 1.0
        loss = torch.FloatTensor([10000.])

        # log description
        desc = "Nodes {} / {} | Edges {} / {} | Density {} | Loss {} | Label {} | ".format(
            init_num_nodes, init_num_nodes,
            init_non_zero_elements, init_non_zero_elements,
            density,
            loss.item(),
            self.label_to_tumor_type[label.item()]
        )
        for label_idx, label_name in self.label_to_tumor_type.items():
            desc += ' ' + label_name + ' {} / {} | '.format(
                round(float(init_probs[int(label_idx)]), 2),
                round(float(init_probs[int(label_idx)]), 2)
            )

        pbar = tqdm(range(self.train_params['num_epochs']), desc=desc, unit='step')
    
        for step in pbar:
            logits, masked_adj, masked_feats = explainer()
            loss = explainer.loss(logits)

            # Compute number of non zero elements in the masked adjacency
            masked_adj = (masked_adj > self.adj_thresh).to(self.device).to(torch.float) * masked_adj
            node_importance = explainer._get_node_feats_mask()
            node_weights = (node_importance > self.node_thresh)  # keep always 10 whatever
            if torch.sum(node_weights).item() <= 10:
                largest_node_indices = node_importance.argsort(descending=True)[:10]
                node_weights = torch.zeros(node_weights.shape, dtype=torch.bool).to(self.device)
                node_weights[largest_node_indices] = True
            masked_feats = masked_feats * torch.stack(masked_feats.shape[-1] * [node_weights], dim=1).unsqueeze(dim=0).to(torch.float)
            probs = torch.nn.Softmax()(logits.cpu().squeeze()).detach().numpy()
            non_zero_elements = (masked_adj != 0).sum()
            density = round(non_zero_elements.item() / init_non_zero_elements.item(), 2)
            num_nodes = torch.sum(masked_feats.sum(dim=-1) != 0.)
            pred_label = torch.argmax(logits, dim=0).squeeze()

            # update description
            desc = "Nodes {} / {} | Edges {} / {} | Density {} | Loss {} | Label {} | ".format(
                num_nodes, init_num_nodes,
                non_zero_elements, init_non_zero_elements,
                density,
                round(loss.item(), 2),
                self.label_to_tumor_type[label.item()]
            )
            for label_idx, label_name in self.label_to_tumor_type.items():
                desc += ' ' + label_name + ' {} / {} | '.format(
                    round(float(probs[int(label_idx)]), 2),
                    round(float(init_probs[int(label_idx)]), 2)
                )

            pbar.set_description(desc)

            # handle early stopping if the labels is changed
            if pred_label.item() == init_pred_label:
                self.adj_explanation = masked_adj
                self.node_feats_explanation = masked_feats
                self.probs_explanation = probs
                self.node_importance = node_importance.cpu().detach().numpy()
            else:
                print('Predicted label changed. Early stopping.')
                break

            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            explainer.optimizer.step()

        # clean up the representation and transform it as networkx object 
        node_idx = (self.node_feats_explanation.squeeze().sum(dim=-1) != 0.).squeeze().cpu()
        adj = self.adj_explanation.squeeze()[node_idx, :]
        adj = adj[:, node_idx]
        feats = self.node_feats_explanation.squeeze()[node_idx, :]
        node_importance = node_importance[node_idx]
        centroids = data[0].ndata['centroid'].squeeze()
        pruned_centroids = centroids[node_idx, :]
        explanation_graph = adj_to_networkx(adj, feats, node_importance=node_importance, threshold=self.model_params['adj_thresh'], centroids=pruned_centroids)

        # build explanation object
        explanation = GraphExplanation(
            self.model_params,
            image,
            image_name,
            init_probs,
            label,
            explanation_graph,
            probs
        )

        return explanation
