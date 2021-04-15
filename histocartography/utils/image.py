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
    return int(last_step)


def pad_image(image, im_h, im_w):
    last_h = get_last_steps(im_h, MASK_SIZE[0], STEP_SIZE[0])
    last_w = get_last_steps(im_w, MASK_SIZE[1], STEP_SIZE[1])
    diff_h = WIN_SIZE[0] - STEP_SIZE[0]
    padt = diff_h // 2
    padb = last_h + WIN_SIZE[0] - im_h
    diff_w = WIN_SIZE[1] - STEP_SIZE[1]
    padl = diff_w // 2
    padr = last_w + WIN_SIZE[1] - im_w
    image = np.pad(image, ((padt, padb), (padl, padr), (0, 0)), 'reflect')
    return image, last_h, last_w


def extract_patches_from_image(image, im_h, im_w):
    x, last_h, last_w = pad_image(image, im_h, im_w)
    sub_patches = []
    coords = []
    # generating subpatches from original
    for row in range(0, last_h, STEP_SIZE[0]):
        for col in range(0, last_w, STEP_SIZE[1]):
            win = x[row:row + WIN_SIZE[0],
                    col:col + WIN_SIZE[1]]
            sub_patches.append(win)
            # left, bottom, right, top
            coords.append([col, row, col + STEP_SIZE[0], row + STEP_SIZE[1]])
    return sub_patches, coords
