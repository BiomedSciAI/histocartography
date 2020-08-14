import torch
import torchvision
from torch import nn
import dgl 

from histocartography.ml.layers.gin_layer import GINLayer
from histocartography.ml.layers.mlp import MLP


class ModelsComponents:
    def __init__(self, config):
        self.experiment_name = config.experiment_name
        self.patch_size = config.patch_size
        self.num_classes = config.num_classes
        self.is_pretrained = config.is_pretrained

        self.model_config = 1
        self._set_default_parameters()
        self.num_features = -1

        # Get model parameters
        if 'resnet' in self.experiment_name:
            self.get_resnet_params()
        elif 'wrn' in self.experiment_name:
            self.get_wrn_params()
        elif 'densenet' in self.experiment_name:
            self.get_densenet_params()

        self.get_embedding_model()

        dropout = config.dropout
        self.get_final_layer(dropout=dropout)

    def _set_default_parameters(self):
        self.block_name = ''
        self.base_filters = -1
        self.depth = -1
        self.width = -1
        self.nb_filters = -1
        self.growth_rate = -1

    def get_resnet_params(self):

        def set_params(block_name='', base_filters=-1, depth=-1):
            self.block_name = block_name
            self.base_filters = base_filters
            self.depth = depth

        # ResNet-29 (3*9 + 2)
        if self.model_config == 1:
            set_params(block_name='Bottleneck', base_filters=16, depth=29)
        elif self.model_config == 2:
            set_params(block_name='Bottleneck', base_filters=32, depth=29)

        # ResNet-56 (6*9 + 2)
        if self.model_config == 3:
            set_params(block_name='Bottleneck', base_filters=16, depth=56)
        elif self.model_config == 4:
            set_params(block_name='Bottleneck', base_filters=32, depth=56)

        # ResNet-110 (12*9 + 2)
        if self.model_config == 5:
            set_params(block_name='Bottleneck', base_filters=16, depth=110)
        elif self.model_config == 6:
            set_params(block_name='Bottleneck', base_filters=32, depth=110)

    def get_wrn_params(self):

        def set_params(depth=-1, width=-1):
            self.depth = depth
            self.width = width

        # WRN-22-(1,2,3)
        if self.model_config == 1:
            set_params(depth=22, width=1)
        elif self.model_config == 2:
            set_params(depth=22, width=2)
        elif self.model_config == 3:
            set_params(depth=22, width=3)

        # WRN-40-(1,2,3)
        elif self.model_config == 4:
            set_params(depth=40, width=1)
        elif self.model_config == 5:
            set_params(depth=40, width=2)
        elif self.model_config == 6:
            set_params(depth=40, width=3)

    def get_densenet_params(self):
        # depth = 22, 28, 34, 40, 46, 52, 58
        # growth_rate = 6, 12, 24

        def set_params(block_name='Bottleneck', depth=-1, growth_rate=-1):
            self.block_name = block_name
            self.depth = depth
            self.growth_rate = growth_rate

        # Growth rate k = 6
        if self.model_config == 1:
            set_params(block_name='Bottleneck', depth=22, growth_rate=6)
        elif self.model_config == 2:
            set_params(block_name='Bottleneck', depth=40, growth_rate=6)
        elif self.model_config == 3:
            set_params(block_name='Bottleneck', depth=58, growth_rate=6)

        # Growth rate k = 12
        elif self.model_config == 4:
            set_params(block_name='Bottleneck', depth=22, growth_rate=12)
        elif self.model_config == 5:
            set_params(block_name='Bottleneck', depth=40, growth_rate=12)
        elif self.model_config == 6:
            set_params(block_name='Bottleneck', depth=58, growth_rate=12)

        # Growth rate k = 24
        elif self.model_config == 7:
            set_params(block_name='Bottleneck', depth=22, growth_rate=24)
        elif self.model_config == 8:
            set_params(block_name='Bottleneck', depth=40, growth_rate=24)
        elif self.model_config == 9:
            set_params(block_name='Bottleneck', depth=58, growth_rate=24)

    def get_embedding_model(self):
        if 'resnet' in self.experiment_name:
            if self.model_config == 18:
                model = torchvision.models.resnet18(
                    pretrained=self.is_pretrained)
                self.num_features = list(model.children())[-1].in_features
                model = torch.nn.Sequential(*(list(model.children())[:-1]))

            elif self.model_config == 34:
                model = torchvision.models.resnet34(
                    pretrained=self.is_pretrained)
                self.num_features = list(model.children())[-1].in_features
                model = torch.nn.Sequential(*(list(model.children())[:-1]))

            elif self.model_config == 50:
                model = torchvision.models.resnet50(
                    pretrained=self.is_pretrained)
                self.num_features = list(model.children())[-1].in_features
                model = torch.nn.Sequential(*(list(model.children())[:-1]))

            elif self.model_config == 101:
                model = torchvision.models.resnet101(
                    pretrained=self.is_pretrained)
                self.num_features = list(model.children())[-1].in_features
                model = torch.nn.Sequential(*(list(model.children())[:-1]))

            elif self.model_config == 152:
                model = torchvision.models.resnet152(
                    pretrained=self.is_pretrained)
                self.num_features = list(model.children())[-1].in_features
                model = torch.nn.Sequential(*(list(model.children())[:-1]))

            else:
                if self.is_pretrained:
                    print(
                        'ERROR: pre-trained models not available for this network. Changing is_pretrained to FALSE.')
                    self.is_pretrained = False

                from preresnet import preresnet
                model = preresnet(
                    block_name=self.block_name,
                    depth=self.depth,
                    base_filters=self.base_filters,
                    num_classes=self.num_classes)
                self.num_features = model.fc.in_features

        elif 'wrn' in self.experiment_name:
            if self.model_config == 50:
                model = torchvision.models.wide_resnet50_2(
                    pretrained=self.is_pretrained)
                self.num_features = list(model.children())[-1].in_features
                model = torch.nn.Sequential(*(list(model.children())[:-1]))

            elif self.model_config == 101:
                model = torchvision.models.wide_resnet101_2(
                    pretrained=self.is_pretrained)
                self.num_features = list(model.children())[-1].in_features
                model = torch.nn.Sequential(*(list(model.children())[:-1]))

            else:
                if self.is_pretrained:
                    print(
                        'ERROR: pre-trained models not available for this network. Changing is_pretrained to FALSE.')
                    self.is_pretrained = False

                from wrn import wrn
                model = wrn(
                    depth=self.depth,
                    width=self.width,
                    num_classes=self.num_classes)
                self.num_features = model.fc.in_features

        elif 'densenet' in self.experiment_name:
            if self.model_config == 121:
                print('Densenet121: TODO')
                exit()

            elif self.model_config == 161:
                print('Densenet121: TODO')
                exit()

            elif self.model_config == 169:
                print('Densenet121: TODO')
                exit()

            else:
                if self.is_pretrained:
                    print(
                        'ERROR: pre-trained models not available for this network. Changing is_pretrained to FALSE.')
                    self.is_pretrained = False

                from densenet import densenet
                model = densenet(
                    depth=self.depth,
                    block_name=self.block_name,
                    growth_rate=self.growth_rate,
                    num_classes=self.num_classes)
                self.num_features = model.fc.in_features

        elif 'vgg' in self.experiment_name:
            if self.model_config == 16:
                model = torchvision.models.vgg16_bn(
                    pretrained=self.is_pretrained)
                classifier = list(model.classifier.children())[:1]
                self.num_features = list(
                    model.classifier.children())[-1].in_features
                model.classifier = nn.Sequential(*classifier)

            elif self.model_config == 19:
                model = torchvision.models.vgg19_bn(
                    pretrained=self.is_pretrained)
                classifier = list(model.classifier.children())[:1]
                self.num_features = list(
                    model.classifier.children())[-1].in_features
                model.classifier = nn.Sequential(*classifier)
        # endif

        if self.is_pretrained:
            for param in model.parameters():
                param.requires_grad = False
        self.embedding_model = model

    def get_final_layer(self, dropout):
        from finallayer import finallayer
        self.classifier_model = finallayer(
            num_classes=self.num_classes,
            in_filters=self.num_features,
            dropout=dropout)


class ModelsStage2:
    def __init__(self, models, config):
        super(ModelsStage2, self).__init__()
        self.embedding_model, self.classifier_model = models
        self.weight_merge = config.weight_merge
        self.num_classes = config.num_classes

    def forward(self, x):
        embedding = self.embedding_model(x)
        embedding = embedding.squeeze(dim=2)
        embedding = embedding.squeeze(dim=2)
        self.dim = embedding.shape[1]

        if self.weight_merge:
            probabilities = self.classifier_model(embedding)
        else:
            probabilities = torch.ones(embedding.shape[0], self.num_classes)

        emb = embedding.repeat(1, self.num_classes)
        prob = probabilities.repeat_interleave(self.dim, dim=1)
        embedding_wt = torch.mean(emb * prob, dim=0)

        return embedding_wt


class TwoLayerGNNCls(torch.nn.Module):

    def __init__(self, input_dimension, num_classes, hidden_dimension=32):
        
        super(TwoLayerGNNCls, self).__init__()
        self.gnn1 = GINLayer(input_dimension, hidden_dimension, 'relu', 0, config=None, verbose=False)
        self.gnn2 = GINLayer(hidden_dimension, hidden_dimension, 'relu', 0, config=None, verbose=False)
        self.cls = MLP(hidden_dimension, hidden_dimension, num_classes, 2)

    def forward(self, x):
        h = x.ndata['h']
        h = self.gnn1(x, h)
        h = self.gnn2(x, h)
        x.ndata['h'] = h 
        h_g = dgl.mean_nodes(x, 'h')
        out = self.cls(h_g)
        return out
