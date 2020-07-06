import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import glob
import time
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_param')                # local, dataT
parser.add_argument('--norm_method')               # macenko_nofit, macenko_fit, vahadane_fit

args = parser.parse_args()
data_param = args.data_param
norm_method = args.norm_method

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if norm_method == 'macenko_nofit':
    from stainNorm_Macenko_nofit import stainingNorm_Macenko_nofit

elif norm_method == 'macenko_fit':
    from stainNorm_Macenko_fit import Normalizer
    target_image_path = './target_image/1231_benign_1.png'
    # ------------------------------------------------------------ Fit normalizer
    img_target_ = Image.open(target_image_path)
    img_target = np.array(img_target_)
    img_target_.close()
    norm_fit = Normalizer()
    norm_fit.fit(img_target)

elif norm_method == 'vahadane_fit':
    from stainNorm_Vahadane import Normalizer
    target_image_path = './target_image/1231_benign_1.png'
    # ------------------------------------------------------------ Fit normalizer
    img_target_ = Image.open(target_image_path)
    img_target = np.array(img_target_)
    img_target_.close()
    norm_fit = Normalizer()
    norm_fit.fit(img_target)


# ----------------------------------------------------------------------------------------------------------------------
# MAIN CODE
# ----------------------------------------------------------------------------------------------------------------------
if data_param == 'local':
    base_path = '/Users/gja/Documents/PhD/histocartography/data/Scan6_7_8_9_10/'
elif data_param == 'dataT':
    base_path = '/dataT/pus/histocartography/Data/PASCALE/'

TUMOR_TYPES = os.listdir(os.path.join(base_path, 'Images'))

print('Tumor types to process: {}'.format(TUMOR_TYPES))

for tumor_type in TUMOR_TYPES:
    print('Start processing tumor type:', tumor_type)

    base_img_path = os.path.join(base_path, 'Images', tumor_type)
    base_save_path = os.path.join(base_path, 'Images_norm')
    create_directory(base_save_path)
    base_save_path = os.path.join(base_save_path, tumor_type)
    create_directory(base_save_path)

    filepaths = glob.glob(os.path.join(base_img_path, '*.png'))
    filepaths.sort()

    for i in tqdm(range(len(filepaths))):

        start_time = time.time()
        filename = os.path.basename(filepaths[i])
        img_rgb = Image.open(filepaths[i])
        img_rgb = np.array(img_rgb).astype(np.uint8)

        if norm_method == 'macenko_nofit':
            normalized = stainingNorm_Macenko_nofit(img_rgb)

        elif norm_method == 'macenko_fit':
            normalized = norm_fit.transform(img_rgb)

        elif norm_method == 'vahadane_fit':
            normalized = norm_fit.transform(img_rgb)

        Image.fromarray(normalized).save(os.path.join(base_save_path, filename))
