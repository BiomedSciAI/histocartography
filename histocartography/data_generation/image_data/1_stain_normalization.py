import numpy as np
from PIL import Image
import os
import glob
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_param')                # local, dataT
# macenko_nofit, macenko_fit, vahadane_fit
parser.add_argument('norm_method')
# benign, pathologicalbenign, udh, adh, fea, dcis, malignant
parser.add_argument('tumor_type')
parser.add_argument('chunk_id')                  # 0, 1, ...

args = parser.parse_args()
data_param = args.data_param
norm_method = args.norm_method
tumor_type = args.tumor_type
chunk_id = int(args.chunk_id)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
# enddef


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


# -----------------------------------------------------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------------------------------------------------
if data_param == 'local':
    base_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/'
elif data_param == 'dataT':
    base_path = '/dataT/pus/histocartography/Data/PASCALE/'

base_img_path = base_path + 'Images/' + tumor_type + '/'
base_save_path = base_path + 'Images_norm/'
create_directory(base_save_path)
base_save_path += tumor_type + '/'
create_directory(base_save_path)

# ---------------------------------------------------------------------------------------------------- Load file paths
n_chunks = 1

filepaths = sorted(glob.glob(base_img_path + '*.png'))

idx = np.array_split(np.arange(len(filepaths)), n_chunks)
idx = idx[chunk_id]

filepaths = [filepaths[x] for x in idx]


for i in range(len(filepaths)):
    start_time = time.time()
    filename = os.path.basename(filepaths[i]).split('.')[0]

    img_rgb_ = Image.open(filepaths[i])
    img_rgb = np.asarray(img_rgb_)
    img_rgb_.close()

    if norm_method == 'macenko_nofit':
        normalized = stainingNorm_Macenko_nofit(img_rgb)

    elif norm_method == 'macenko_fit':
        normalized = norm_fit.transform(img_rgb)

    elif norm_method == 'vahadane_fit':
        normalized = norm_fit.transform(img_rgb)
    # endif

    Image.fromarray(normalized).save(base_save_path + filename + '.png')
    print(
        '#',
        i,
        ' : ',
        filename,
        ' time: ',
        round(
            time.time() -
            start_time,
            2),
        's')
# endfor
