import time
import numpy as np
import torch
from tqdm import tqdm

from ..ml.layers.constants import GNN_NODE_FEAT_IN
from ..dataloader.constants import LABEL_TO_TUMOR_TYPE
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
        init_logits = self.model(data).cpu().detach()
        init_probs = torch.nn.Softmax()(init_logits).numpy().squeeze()
        pred_label = np.argmax(init_logits, axis=0)

        # print('Pred label {} | Groud truth label {}'.format(
        #     pred_label,
        #     label)
        # )

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
        # Init training stats
        init_non_zero_elements = (adj != 0).sum()
        density = 1.0
        loss = torch.FloatTensor([10000.])
        desc = "Non-zero: {} / {} | Density: {} | Loss: {} | Label: {} | N: {} / {} | B: {} / {} | ATY: {} / {} | DCIS: {} / {} | I: {} / {}".format(
            init_non_zero_elements,
            init_non_zero_elements,
            density,
            loss.item(), 
            LABEL_TO_TUMOR_TYPE[str(label.item())],
            round(float(init_probs[0]), 1), round(float(init_probs[0]), 1),
            round(float(init_probs[1]), 1), round(float(init_probs[1]), 1),
            round(float(init_probs[3]), 1), round(float(init_probs[3]), 1),
            round(float(init_probs[4]), 1), round(float(init_probs[4]), 1),
            round(float(init_probs[2]), 1), round(float(init_probs[2]), 1)
        )
        pbar = tqdm(range(self.train_params['num_epochs']), desc=desc, unit='step')
    
        for step in pbar:
            logits, masked_adj, masked_feats = explainer()
            loss = explainer.loss(logits)

            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            loss.backward()
            explainer.optimizer.step()

            # Compute number of non zero elements in the masked adjacency
            probs = torch.nn.Softmax()(logits.cpu().squeeze()).detach().numpy()
            non_zero_elements = (masked_adj != 0).sum()
            density = round(non_zero_elements.item() / init_non_zero_elements.item(), 2)
            desc = 'Non-zero: {} / {} | Density {} | Loss: {}'.format(
                non_zero_elements,
                init_non_zero_elements,
                density,
                round(loss.item(), 2))
            
            desc = "Non-zero: {} / {} | Density: {} | Loss: {} | Label: {} | N: {} / {} | B: {} / {} | ATY: {} / {} | DCIS: {} / {} | I: {} / {}".format(
                non_zero_elements,
                init_non_zero_elements,
                density,
                round(loss.item(), 2),
                LABEL_TO_TUMOR_TYPE[str(label.item())],
                round(float(probs[0]), 1), round(float(init_probs[0]), 1),
                round(float(probs[1]), 1), round(float(init_probs[1]), 1),
                round(float(probs[3]), 1), round(float(init_probs[3]), 1),
                round(float(probs[4]), 1), round(float(init_probs[4]), 1),
                round(float(probs[2]), 1), round(float(init_probs[2]), 1)
        )

            pbar.set_description(desc)

            # print('Loss: {} | Mask density: {} | Prediction: {}'.format(
            # loss.item(),
            # self.mask_density().item(),
            # self.label == torch.argmax(pred).item()
            # ))
            # print('Prediction:', pred)

        ypred, _, _ = explainer()

        # print("Training time: {} with density {} | with prediction {}".format(
        #     time.time() - begin_time,
        #     explainer.mask_density().item(),
        #     ypred)
        # )

        return masked_adj.squeeze(), masked_feats
