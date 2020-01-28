import numpy as np
from PIL import Image
import os
import glob
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("norm_method")               # macenko_nofit, macenko_fit, vahadane_fit
args = parser.parse_args()
norm_method = args.norm_method

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
#enddef

if norm_method == 'macenko_nofit':
    from stainNorm_Macenko_nofit import stainingNorm_Macenko_nofit

elif norm_method == 'macenko_fit':
    from stainNorm_Macenko_fit import Normalizer
    target_image_path = '/Users/pus/Desktop/pascale/troi/0_benign/1231_benign_1.png'
    norm_fit = ''

elif norm_method == 'vahadane_fit':
    from stainNorm_Vahadane import Normalizer
    target_image_path = '/Users/pus/Desktop/pascale/troi/0_benign/1231_benign_1.png'
    norm_fit = ''


#-----------------------------------------------------------------------------------------------------------------------
# MAIN CODE
#-----------------------------------------------------------------------------------------------------------------------

tumor_type = '6_malignant'     # 0_benign, 1_pathological_benign, 2_udh, 3_adh, 4_fea, 5_dcis, 6_malignant

base_img_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/Images/' + tumor_type + '/'
base_save_path = '/Users/pus/Desktop/Projects/Data/Histocartography/PASCALE/Images_norm/' + tumor_type + '/'
create_directory(base_save_path)

filepaths = glob.glob(base_img_path + '*.png')
filepaths.sort()

#for i in range(len(filepaths)):

img_target_ = Image.open(target_image_path)
img_target = np.array(img_target_)
img_target_.close()
norm_fit = Normalizer()
norm_fit.fit(img_target)


n = 6

for i in range(0, n*int(len(filepaths)/6)):
    if i < (n-1) * int(len(filepaths)/6):
        continue

    start_time = time.time()
    filename = os.path.basename(filepaths[i]).split('.')[0]

    img_rgb_ = Image.open(filepaths[i])
    img_rgb = np.asarray(img_rgb_)
    img_rgb_.close()

    if norm_method == 'macenko_nofit':
        normalized = stainingNorm_Macenko_nofit(img_rgb)

    elif norm_method == 'macenko_fit':
        '''
        if i == 0:
            img_target_ = Image.open(target_image_path)
            img_target = np.array(img_target_)
            img_target_.close()
            norm_fit = Normalizer()
            norm_fit.fit(img_target)
        #'''

        normalized = norm_fit.transform(img_rgb)

    elif norm_method == 'vahadane_fit':
        if i == 0:
            img_target_ = Image.open(target_image_path)
            img_target = np.array(img_target_)
            img_target_.close()
            norm_fit = Normalizer()
            norm_fit.fit(img_target)

        normalized = norm_fit.transform(img_rgb)
    #endif

    Image.fromarray(normalized).save(base_save_path + filename + '.png')
    print('#', i, ' : ', filename, ' time: ', round(time.time() - start_time, 2), 's')

#endfor















