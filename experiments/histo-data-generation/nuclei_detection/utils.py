import cv2
import numpy as np
import random
import colorsys
from skimage.feature import greycomatrix, greycoprops
from skimage.filters.rank import entropy as Entropy
from skimage.morphology import disk
import scipy
import h5py
import os
from skimage.measure import regionprops


def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
# enddef


def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)
# enddef


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]
# enddef


def random_colors(n, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
# enddef


def visualize_instances(mask, canvas=None, color=None):
    """
    Args:
        mask: array of NW
    Return:
        Image with the instance overlaid
    """

    canvas = np.full(mask.shape + (3,), 200,
                     dtype=np.uint8) if canvas is None else np.copy(canvas)

    insts_list = list(np.unique(mask))
    insts_list.remove(0)  # remove background

    inst_colors = random_colors(len(insts_list))
    inst_colors = np.array(inst_colors) * 255

    for idx, inst_id in enumerate(insts_list):
        inst_color = color[idx] if color is not None else inst_colors[idx]
        inst_map = np.array(mask == inst_id, np.uint8)
        y1, y2, x1, x2 = bounding_box(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= mask.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= mask.shape[0] - 1 else y2
        inst_map_crop = inst_map[y1:y2, x1:x2]
        inst_canvas_crop = canvas[y1:y2, x1:x2]
        _, contours, _ = cv2.findContours(
            inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(inst_canvas_crop, contours[0], -1, inst_color, 2)
        canvas[y1:y2, x1:x2] = inst_canvas_crop
    return canvas
# enddef


def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nuclei has smaller ID
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label

    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred
# enddef


def extract_centroid(mask):
    insts_list = list(np.unique(mask))
    insts_list.remove(0)  # remove background
    centroids = []

    for idx, inst_id in enumerate(insts_list):
        inst_map = np.array(mask == inst_id, np.uint8)

        # get bounding box for each nuclei
        y1, y2, x1, x2 = bounding_box(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= mask.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= mask.shape[0] - 1 else y2
        nuclei_mask = inst_map[y1:y2, x1:x2]

        _, contours, _ = cv2.findContours(
            nuclei_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        M = cv2.moments(contour)

        if M["m00"] == 0:
            return np.array([])

        if M["m00"] != 0:
            cX = (M["m10"] / M["m00"])
            cY = (M["m01"] / M["m00"])
            centroid = [x1 + cX, y1 + cY]
            centroids.append(centroid)
    # endfor

    centroids = np.vstack(centroids)
    return centroids
# enddef


def extract_feat(img, mask):
    img = np.full(mask.shape + (3,), 200,
                  dtype=np.uint8) if img is None else np.copy(img)

    insts_list = list(np.unique(mask))
    insts_list.remove(0)  # remove background

    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    node_feat = []
    entropy = Entropy(img_g, disk(3))

    for idx, inst_id in enumerate(insts_list):
        inst_map = np.array(mask == inst_id, np.uint8)
        nuc_feat = []

        # get bounding box for each nuclei
        y1, y2, x1, x2 = bounding_box(inst_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= mask.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= mask.shape[0] - 1 else y2
        nuclei_mask = inst_map[y1:y2, x1:x2]
        nuclei_img_g = img_g[y1:y2, x1:x2]
        nuclei_entropy = entropy[y1:y2, x1:x2]

        background_px = np.array(nuclei_img_g[nuclei_mask == 0])
        foreground_px = np.array(nuclei_img_g[nuclei_mask > 0])

        # Morphological features (mean_fg, diff, var, skew)
        mean_bg = background_px.sum() / (np.size(background_px) + 1.0e-8)
        mean_fg = foreground_px.sum() / (np.size(foreground_px) + 1.0e-8)
        diff = abs(mean_fg - mean_bg)
        var = np.var(foreground_px)
        skew = scipy.stats.skew(foreground_px)

        # Textural features (gray level co-occurence matrix)
        glcm = greycomatrix(nuclei_img_g * nuclei_mask, [1], [0])
        filt_glcm = glcm[1:, 1:, :, :]  # Filter out the first row and column
        glcm_contrast = greycoprops(filt_glcm, prop='contrast')
        glcm_contrast = glcm_contrast[0, 0]
        glcm_dissimilarity = greycoprops(filt_glcm, prop='dissimilarity')
        glcm_dissimilarity = glcm_dissimilarity[0, 0]
        glcm_homogeneity = greycoprops(filt_glcm, prop='homogeneity')
        glcm_homogeneity = glcm_homogeneity[0, 0]
        glcm_energy = greycoprops(filt_glcm, prop='energy')
        glcm_energy = glcm_energy[0, 0]
        glcm_ASM = greycoprops(filt_glcm, prop='ASM')
        glcm_ASM = glcm_ASM[0, 0]

        mean_entropy = cv2.mean(nuclei_entropy, mask=nuclei_mask)[0]

        _, contours, _ = cv2.findContours(
            nuclei_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]

        num_vertices = len(contour)
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            hull_area += 1
        solidity = float(area) / hull_area
        if num_vertices > 4:
            centre, axes, orientation = cv2.fitEllipse(contour)
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
        else:
            orientation = 0
            majoraxis_length = 1
            minoraxis_length = 1
        perimeter = cv2.arcLength(contour, True)
        eccentricity = np.sqrt(1 - (minoraxis_length / majoraxis_length) ** 2)

        nuc_feat.append(mean_fg)
        nuc_feat.append(diff)
        nuc_feat.append(var)
        nuc_feat.append(skew)
        nuc_feat.append(mean_entropy)
        nuc_feat.append(glcm_dissimilarity)
        nuc_feat.append(glcm_homogeneity)
        nuc_feat.append(glcm_energy)
        nuc_feat.append(glcm_ASM)
        nuc_feat.append(eccentricity)
        nuc_feat.append(area)
        nuc_feat.append(majoraxis_length)
        nuc_feat.append(minoraxis_length)
        nuc_feat.append(perimeter)
        nuc_feat.append(solidity)
        nuc_feat.append(orientation)

        features = np.hstack(nuc_feat)
        node_feat.append(features)
    # endfor

    node_feat = np.vstack(node_feat)
    return node_feat
# endfor


def save_instance_map_h5(h5_filename, inst_map, data_dtype='float32'):
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset(
        'detected_instance_map',
        data=inst_map,
        dtype=data_dtype)
    h5_fout.close()
# enddef


def save_class_type_h5(
        h5_filename,
        pred_type,
        inst_type,
        data_dtype='float32'):
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset('detected_type', data=pred_type, dtype=data_dtype)
    h5_fout.create_dataset('instance_types', data=inst_type, dtype=data_dtype)
    h5_fout.close()
# enddef


def save_centroid_h5(h5_filename, centroid, img_dim, data_dtype='float32'):
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset(
        'instance_centroid_location',
        data=centroid,
        dtype=data_dtype)
    h5_fout.create_dataset('image_dimension', data=img_dim, dtype='int32')
    h5_fout.close()
# enddef
