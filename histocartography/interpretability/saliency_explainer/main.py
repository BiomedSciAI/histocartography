import argparse
from warnings import simplefilter, filterwarnings
import torch.nn as nn 
import torch 
from PIL import Image

simplefilter(action='ignore', category=FutureWarning)
filterwarnings(action='ignore', category=DeprecationWarning)
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data_param', choices=['local', 'dataT'], default='local', required=True)
parser.add_argument('--explainer_type', required=True)
args = parser.parse_args()

if args.data_param == 'local':
    base_path = '/Users/pus/Desktop/Projects/Code/histocartography/'
    base_image_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/Images_norm/dcis/'
else:
    base_path = '/dataT/pus/histocartography/interpretability/histocartography/'
    base_image_path = '/dataT/pus/histocartography/Data/BRACS_L/Images_norm/dcis/'


sys.path.append(base_path)
sys.path.append(base_path + 'histocartography/')
sys.path.append(base_path + 'histocartography/baseline/cnn_baselines/')
sys.path.append(base_path + 'histocartography/interpretability/')
sys.path.append(base_path + 'histocartography/interpretability/saliency_explainer/')
sys.path.append(base_path + 'histocartography/baseline/cnn_baselines/')


import os
# from cam_explainer import *
from histocartography.interpretability.saliency_explainer.image_gradcam_explainer import ImageGradCAMExplainer
# from gradcampp_explainer import *
# from smoothgradcampp_explainer import *
# from scorecam_explainer import *
# from sscam_explainer import *


# redefine ResNet without re-using ReLU activations 

from torchvision.models.resnet import _resnet, conv3x3, conv1x1


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        # Added another relu here
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        # Modified to use relu2
        out = self.relu2(out)

        return out

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze(dim=2)


def get_model():

    save_to_mlflow = True
    embedding_model = torch.load(model_path + 'embedding_model_best_f1.pt', map_location=torch.device('cpu'))
    resnet50_model = resnet50(pretrained=True)
    resnet50_model = nn.Sequential(*list(resnet50_model.children())[:-1])
    resnet50_model.load_state_dict(embedding_model.state_dict())
    classification_model = torch.load(model_path + 'classification_model_best_f1.pt', map_location=torch.device('cpu'))
    squeeze = Squeeze()
    model = torch.nn.Sequential(*resnet50_model.children(),
                                squeeze,
                                squeeze,
                                *classification_model.children())
    for param in model.parameters():
        param.requires_grad = True

    if save_to_mlflow:
        import mlflow.pytorch 
        mlflow.pytorch.log_model(model, 'model_5_classes')

    return model


def get_explainer(explainer_type):
    # if explainer_type == 'cam':
    #     explainer = CAMGNNExplainer(model=model, config=None)

    if explainer_type == 'gradcam':

        config = {
            "explanation_type": "saliency_explainer.image_deeplift_explainer",
            "model_params": {
                "model_type": "cnn_model",
                "class_split": "benignVSpathologicalbenign+udhVSadh+feaVSdcisVSmalignant"
            }
        }

        explainer = ImageGradCAMExplainer(model=model, config=config, cuda=True)

    # elif explainer_type == 'gradcampp':
    #     explainer = GradCAMPPGNNExplainer(model=model, config=None)

    # elif explainer_type == 'smoothgradcampp':
    #     explainer = SmoothGradCAMPPGNNExplainer(model=model, config=None)

    # elif explainer_type == 'scorecam':
    #     explainer = ScoreCAMGNNExplainer(model=model, config=None)

    # elif explainer_type == 'sscam':
    #     explainer = SSCAMGNNExplainer(model=model, config=None)

    else:
        print('ERROR')

    return explainer


if __name__ == '__main__':

    # Paths
    #model_path = './resnet50_single_scale_10x_ps128_bs64_lr0.0001_spie/4/'
    model_path = 'C5/'

    base_savepath = './explanations/'
    create_directory(base_savepath)

    # Constants
    tumor_type = ['benign', 'pathologicalbenign', 'udh', 'adh', 'fea', 'dcis', 'malignant']
    tumor_label = [0, 1, 1, 2, 2, 3, 4]

    # Explainer method
    explainer_type = args.explainer_type

    # Test image
    image_path = base_image_path + '283_dcis_1.png'
    label = tumor_label[tumor_type.index(os.path.basename(image_path).split('_')[1])]

    # Get
    model = get_model()
    explainer = get_explainer(explainer_type=explainer_type)

    # Process
    img = Image.open(image_path, mode='r').convert('RGB')
    heatmap = explainer.explain(data=[[img], ['image']], label=label)



