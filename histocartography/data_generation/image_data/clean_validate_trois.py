'''
TO BE USED LATER. (Date: 7-1-2020)
IDEA: For TRoIs larger than a specified size, generate random crops of maximum size
'''

import numpy as np
from PIL import Image
import glob
import os

base_path = '/Users/pus/Desktop/pascale/trois/'
dir_list = glob.glob(base_path + '*/')
max_img_h = 5000
max_img_w = 5000
dimension_threshold = max_img_h * max_img_w  # Upper limit on the dimension of an image in terms of pixels

def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)

save_large_file_path = base_path + 'large_images/'
create_directory(save_large_file_path)

tumor_types = ['0_benign', '1_pathological_benign', '2_udh', '3_adh', '4_fea', '5_dcis', '6_malignant']
for tt in tumor_types:
    create_directory(save_large_file_path + tt)


for dir in dir_list:
    img_list = glob.glob(dir + '*.png')
    img_list.sort()

    for i in range(len(img_list)):
        img_ = Image.open(img_list[i])
        img = np.array(img_)
        img_.close()
        (h, w, c) = img.shape

        if h*w > dimension_threshold:
            filename = os.path.basename(img_list[i]).split('.')[0]
            dirname = os.path.dirname(img_list[i]) + '/'

            n = int(round(h*w/dimension_threshold, 0))

            if (h - max_img_h) % n == 0:
                increment = int((h - max_img_h)/n) - 1
            else:
                increment = int((h - max_img_h) / n)
            h_list = np.arange(0, h-max_img_h, increment)

            if (w - max_img_w) % n == 0:
                increment = int((w - max_img_w) / n) - 1
            else:
                increment = int((w - max_img_w) / n)
            w_list = np.arange(0, w - max_img_w, increment)

            print(h, w, n)
            print(h_list)
            print(w_list)

            np.random.seed(i)
            for j in range(n):
                h_ = np.random.randint(low=h_list[j], high=h_list[j+1], size=1)[0]
                w_ = np.random.randint(low=w_list[j], high=w_list[j+1], size=1)[0]

                img_cropped = img[h_:h_+max_img_h, w_:w_+max_img_w, :]
                Image.fromarray(img_cropped).save(dirname + filename + '_' + str(j) + '.png')
            #endfor

            dirname = dirname.split('/')[-2]
            os.rename(img_list[i], save_large_file_path + dirname + '/' + filename + '.png')





