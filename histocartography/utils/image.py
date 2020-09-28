import math 
import numpy as np 
import cv2
import os
import glob 

STEP_SIZE = [164, 164]
MASK_SIZE = [164, 164]
WIN_SIZE = [256, 256]


def get_last_steps(length, msk_size, step_size):
    nr_step = math.ceil((length - msk_size) / step_size)
    last_step = (nr_step + 1) * step_size
    return int(last_step), int(nr_step + 1)


def pad_image(image, im_h, im_w):
    last_h, nr_step_h = get_last_steps(im_h, MASK_SIZE[0], STEP_SIZE[0])
    last_w, nr_step_w = get_last_steps(im_w, MASK_SIZE[1], STEP_SIZE[1])
    diff_h = WIN_SIZE[0] - STEP_SIZE[0]
    padt = diff_h // 2
    padb = last_h + WIN_SIZE[0] - im_h
    diff_w = WIN_SIZE[1] - STEP_SIZE[1]
    padl = diff_w // 2
    padr = last_w + WIN_SIZE[1] - im_w
    image = np.pad(image, ((padt, padb), (padl, padr), (0, 0)), 'reflect')
    return image, last_h, last_w, nr_step_h, nr_step_w


def extract_patches_from_image(image, im_h, im_w):
    x, last_h, last_w, nr_step_h, nr_step_w = pad_image(image, im_h, im_w)
    sub_patches = []
    # generating subpatches from original
    for row in range(0, last_h, STEP_SIZE[0]):
        for col in range (0, last_w, STEP_SIZE[1]):
            win = x[row:row+WIN_SIZE[0], 
                    col:col+WIN_SIZE[1]]
            sub_patches.append(win)
    return sub_patches, nr_step_h, nr_step_w


def load_images(data_path):
    all_images = []
    if os.path.isfile(data_path) and data_path.endswith('.png'):
        image = cv2.imread(data_path)  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_images.append((image, data_path.split('/')[-1]))
    else:
        for file in glob.glob(os.path.join(data_path, "*.png")):
            image = cv2.imread(os.path.join(data_path, file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            all_images.append((image, file))
    return all_images 
