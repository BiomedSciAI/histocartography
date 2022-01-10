import math
import numpy as np
import cv2


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


def augment_in_hsv(rgb_img):
    rgb_img = np.array(rgb_img)
    dim = rgb_img.shape
    out = np.empty((dim[0], dim[1], dim[2]))

    # transform to HSV space
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    hsv_img = np.array(hsv_img, dtype=np.float64)
    hsv_img[:, :, 0] = ((hsv_img[:, :, 0].astype(np.float32) / 180.0) * 255.0)

    # sample random perturbations for each channel
    r_h = round(np.random.uniform(0.75, 1.25), 2)
    r_s = round(np.random.uniform(0.70, 1.30), 2)
    r_v = round(np.random.uniform(0.9, 1.10), 2)

    # perturb each channel
    temp = np.empty((dim[0], dim[1], dim[2]))
    temp[:, :, 0] = hsv_img[:, :, 0] * r_h
    temp[:, :, 1] = hsv_img[:, :, 1] * r_s
    temp[:, :, 2] = hsv_img[:, :, 2] * r_v

    temp[:, :, 0] = ((np.mod(temp[:, :, 0], 255).astype(np.float32) / 255.0) * (360 / 2))
    temp[:, :, 1] = np.clip(temp[:, :, 1], 0, 255)
    temp[:, :, 2] = np.clip(temp[:, :, 2], 0, 255)

    temp = temp.astype(np.uint8)
    out[:, :, :] = cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)

    out = out.astype(np.uint8)
    return out
