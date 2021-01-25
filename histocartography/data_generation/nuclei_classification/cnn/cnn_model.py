import torch
from torch import nn
import resnet
import copy

class ModelComponents:
    def __init__(self, config, args):
        self.config = config
        self.args = args

        # Get number of classes
        self.nuclei_types = copy.deepcopy(self.config.nuclei_types)
        idx = self.nuclei_types.index('NA')     # Remove nuclei type = 'NA'
        del self.nuclei_types[-idx]
        self.num_classes = len(self.nuclei_types)

        self.get_embedding_model()
        self.get_classification_model()

    def get_embedding_model(self):
        modelA = torch.nn.DataParallel(resnet.__dict__[self.args.arch]())
        if eval(self.args.pretrained):
            checkpoint = torch.load(self.config.pretrained_model_path + self.args.arch + '.th', map_location=self.config.device)
            modelA.load_state_dict(checkpoint['state_dict'], strict=True)

        # Delete last layer from model
        modelB = resnet.__dict__[self.args.arch]()
        modelB.linear = nn.Identity()

        # Delete last layer from state_dict
        state_dictA = modelA.state_dict()
        state_dict_filt = {k: v for k, v in state_dictA.items() if 'linear' not in k}

        # Correct for key mismatch
        state_dictB = copy.deepcopy(state_dict_filt)
        for key in state_dict_filt:
            keyB = key.replace('module.', '')
            state_dictB[keyB] = state_dictB.pop(key)

        # Load state_dict
        modelB.load_state_dict(state_dictB)

        self.embedding_model = modelB
        self.embedding_features = modelB.num_features

    def get_classification_model(self):
        self.classification_model = classification_layer(
            num_classes=self.num_classes, in_filters=self.embedding_features, dropout=self.args.dropout)


class ClassificationLayer(nn.Module):
    def __init__(self, num_classes, in_filters, dropout):
        super(ClassificationLayer, self).__init__()
        self.dropout = dropout
        self.linear1 = nn.Linear(in_features=in_filters, out_features=32)
        self.linear2 = nn.Linear(in_features=32, out_features=num_classes)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        if self.dropout != 0:
            x = self.drop_out(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def classification_layer(**kwargs):
    return ClassificationLayer(**kwargs)
