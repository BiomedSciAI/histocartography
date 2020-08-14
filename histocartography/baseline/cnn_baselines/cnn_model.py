import torch
import torchvision
from torch import nn

class ModelComponents:
    def __init__(self, config):
        self.encoder = config.encoder
        self.num_classes = config.num_classes
        self.dropout = config.dropout
        self.magnifications = config.magnifications

        self.get_embedding_model()
        self.get_classification_model()


    def get_embedding_model(self):
        if 'resnet' in self.encoder:
            if '18' in self.encoder:
                model = torchvision.models.resnet18(pretrained=True)
                num_features = list(model.children())[-1].in_features
                model = torch.nn.Sequential(*(list(model.children())[:-1]))

            elif '34' in self.encoder:
                model = torchvision.models.resnet34(pretrained=True)
                num_features = list(model.children())[-1].in_features
                model = torch.nn.Sequential(*(list(model.children())[:-1]))

            elif '50' in self.encoder:
                model = torchvision.models.resnet50(pretrained=True)
                num_features = list(model.children())[-1].in_features
                model = torch.nn.Sequential(*(list(model.children())[:-1]))

            elif '101' in self.encoder:
                model = torchvision.models.resnet101(pretrained=True)
                num_features = list(model.children())[-1].in_features
                model = torch.nn.Sequential(*(list(model.children())[:-1]))

            elif '152' in self.encoder:
                model = torchvision.models.resnet152(pretrained=True)
                num_features = list(model.children())[-1].in_features
                model = torch.nn.Sequential(*(list(model.children())[:-1]))

            else:
                print('ERROR: Select from Resnet: 34, 50, 101, 152')

        elif 'vgg' in self.encoder:
            if '16' in self.encoder:
                model = torchvision.models.vgg16_bn(pretrained=True)
                classifier = list(model.classifier.children())[:1]
                num_features = list(model.classifier.children())[-1].in_features
                model.classifier = nn.Sequential(*classifier)

            elif '19' in self.encoder:
                model = torchvision.models.vgg19_bn(pretrained=True)
                classifier = list(model.classifier.children())[:1]
                num_features = list(model.classifier.children())[-1].in_features
                model.classifier = nn.Sequential(*classifier)

        self.embedding_model = model
        self.embedding_features = num_features


    def get_classification_model(self):
        self.classification_model = classification_layer(num_classes=self.num_classes,
                                                 in_filters=len(self.magnifications) * self.embedding_features,
                                                 dropout=self.dropout)


class ClassificationLayer(nn.Module):
    def __init__(self, num_classes, in_filters, dropout):
        super(ClassificationLayer, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features=in_filters, out_features=num_classes)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        if self.dropout != 0:
            x = self.drop_out(x)
        return x


def classification_layer(**kwargs):
    return ClassificationLayer(**kwargs)