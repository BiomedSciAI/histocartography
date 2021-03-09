import torch
from torch import nn
import torchvision


def get_encoding_model(encoder, mode):
    if 'resnet' in encoder:
        if '34' in encoder:
            model = torchvision.models.resnet34(pretrained=True)
            num_features = list(model.children())[-1].in_features
            model = torch.nn.Sequential(*(list(model.children())[:-1]))

        elif '50' in encoder:
            model = torchvision.models.resnet50(pretrained=True)
            num_features = list(model.children())[-1].in_features
            model = torch.nn.Sequential(*(list(model.children())[:-1]))

        elif '101' in encoder:
            model = torchvision.models.resnet101(pretrained=True)
            num_features = list(model.children())[-1].in_features
            model = torch.nn.Sequential(*(list(model.children())[:-1]))

        elif '152' in encoder:
            model = torchvision.models.resnet152(pretrained=True)
            num_features = list(model.children())[-1].in_features
            model = torch.nn.Sequential(*(list(model.children())[:-1]))

        else:
            print('ERROR: Select from Resnet: 34, 50, 101, 152')

    elif 'vgg' in encoder:
        if '16' in encoder:
            model = torchvision.models.vgg16_bn(pretrained=True)
            classifier = list(model.classifier.children())[:1]
            num_features = list(model.classifier.children())[-1].in_features
            model.classifier = nn.Sequential(*classifier)

        elif '19' in encoder:
            model = torchvision.models.vgg19_bn(pretrained=True)
            classifier = list(model.classifier.children())[:1]
            num_features = list(model.classifier.children())[-1].in_features
            model.classifier = nn.Sequential(*classifier)
    # endif

    if 'vae' in mode:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False

    return model, num_features
# enddef
