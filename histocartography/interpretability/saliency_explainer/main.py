import argparse
from warnings import simplefilter, filterwarnings
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
from cam_explainer import *
from gradcam_explainer import *
from gradcampp_explainer import *
from smoothgradcampp_explainer import *
from scorecam_explainer import *
from sscam_explainer import *

def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze(dim=2)


def get_model():
    embedding_model = torch.load(model_path + 'embedding_model_best_f1.pt', map_location=torch.device('cpu'))
    classification_model = torch.load(model_path + 'classification_model_best_f1.pt', map_location=torch.device('cpu'))
    squeeze = Squeeze()
    model = torch.nn.Sequential(*embedding_model.children(),
                                squeeze,
                                squeeze,
                                *classification_model.children())
    return model


def get_explainer(explainer_type):
    if explainer_type == 'cam':
        explainer = CAMGNNExplainer(model=model, config=None)

    elif explainer_type == 'gradcam':
        explainer = GradCAMGNNExplainer(model=model, config=None)

    elif explainer_type == 'gradcampp':
        explainer = GradCAMPPGNNExplainer(model=model, config=None)

    elif explainer_type == 'smoothgradcampp':
        explainer = SmoothGradCAMPPGNNExplainer(model=model, config=None)

    elif explainer_type == 'scorecam':
        explainer = ScoreCAMGNNExplainer(model=model, config=None)

    elif explainer_type == 'sscam':
        explainer = SSCAMGNNExplainer(model=model, config=None)

    else:
        print('ERROR')

    return explainer


if __name__ == '__main__':

    # Paths
    #model_path = './resnet50_single_scale_10x_ps128_bs64_lr0.0001_spie/4/'
    model_path = './C7/'

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
    heatmap = explainer.explain(data=img, label=label)



