
import cv2
import math
import colorsys
import numpy as np
import matplotlib.pyplot as plt
import itertools
import glob
import os
import shutil
import numpy as np
import time
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from skimage.morphology import remove_small_objects, watershed
import skimage
from skimage.morphology import remove_small_holes, disk
from skimage.filters import rank, threshold_otsu
from scipy import ndimage
from scipy.ndimage.morphology import (binary_dilation, binary_erosion, binary_closing)


####
def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image.

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`
        
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


####
def crop_to_shape(x, y, data_format="NCHW"):
    """Centre crop x so that x has shape of y. y dims must be smaller than x dims.

    Args:
        x: input array
        y: array with desired shape.

    """
    assert (
        y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    ), "Ensure that y dimensions are smaller than x dimensions!"

    x_shape = x.size()
    y_shape = y.size()
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])
    return crop_op(x, crop_shape, data_format)


####
def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)

####
def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out 
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

####
def cropping_center(x, crop_shape, batch=False):   
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:,h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]        
    return x

####
def rm_n_mkdir(dir_path):
    if (os.path.isdir(dir_path)):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

####
def get_files(data_dir_list, data_ext):
    """
    Given a list of directories containing data with extention 'data_ext',
    generate a list of paths for all files within these directories
    """

    data_files = []
    for sub_dir in data_dir_list:
        files = glob.glob(sub_dir + '/*'+ data_ext)
        data_files.extend(files)

    return data_files

####
def get_inst_centroid(inst_map):
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]: # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [(inst_moment["m10"] / inst_moment["m00"]),
                         (inst_moment["m01"] / inst_moment["m00"])]
        inst_centroid_list.append(inst_centroid)
    return np.array(inst_centroid_list)



def class_colour(class_value):
    """
    Generate RGB colour for overlay based on class id
    Args:
        class_value: integer denoting the class of object  
    """
    if class_value == 0:
        return 0, 0, 0  # black (background)
    if class_value == 1:
        return 255, 0, 0  # red
    elif class_value == 2:
        return 0, 255, 0  # green
    elif class_value == 3:
        return 0, 0, 255  # blue
    elif class_value == 4:
        return 255, 255, 0  # yellow
    elif class_value == 5:
        return 255, 165, 0  # orange
    elif class_value == 6:
        return 0, 255, 255  # cyan
    else:
        raise Exception(
            'Currently, overlay_segmentation_results() only supports up to 6 classes.')
####

def visualize_instances(input_image, predict_instance, predict_type=None, line_thickness=2):
    """
    Overlays segmentation results on image as contours
    Args:
        input_image: input image
        predict_instance: instance mask with unique value for every object
        predict_type: type mask with unique value for every class
        line_thickness: line thickness of contours
    Returns:
        overlay: output image with segmentation overlay as contours
    """
   
    overlay = np.copy((input_image).astype(np.uint8))

    if predict_type is not None:
        type_list = list(np.unique(predict_type))  # get list of types
        type_list.remove(0)  # remove background
    else:
        type_list = [4]  # yellow

    for iter_type in type_list:
        if predict_type is not None:
            label_map = (predict_type == iter_type) * predict_instance
        else:
            label_map = predict_instance
        instances_list = list(np.unique(label_map))  # get list of instances
        instances_list.remove(0)  # remove background
        contours = []
        for inst_id in instances_list:
            instance_map = np.array(
                predict_instance == inst_id, np.uint8)  # get single object
            y1, y2, x1, x2 = bounding_box(instance_map)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= predict_instance.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= predict_instance.shape[0] - 1 else y2
            inst_map_crop = instance_map[y1:y2, x1:x2]
            contours_crop = cv2.findContours(
                inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            index_correction = np.asarray([[[[x1, y1]]]])
            for i in range(len(contours_crop[0])):
                contours.append(
                    list(np.asarray(contours_crop[0][i].astype('int32')) + index_correction))
        contours = list(itertools.chain(*contours))
        cv2.drawContours(overlay, np.asarray(contours), -1,
                         class_colour(iter_type), line_thickness)
    return overlay
####

def gen_figure(imgs_list, titles, fig_inch, shape=None,
                share_ax='all', show=False, colormap=plt.get_cmap('jet')):

    num_img = len(imgs_list)
    if shape is None:
        ncols = math.ceil(math.sqrt(num_img))
        nrows = math.ceil(num_img / ncols)
    else:
        nrows, ncols = shape

    # generate figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                        sharex=share_ax, sharey=share_ax)
    axes = [axes] if nrows == 1 else axes

    # not very elegant
    idx = 0
    for ax in axes:
        for cell in ax:
            cell.set_title(titles[idx])
            cell.imshow(imgs_list[idx], cmap=colormap)
            cell.tick_params(axis='both', 
                            which='both', 
                            bottom='off', 
                            top='off', 
                            labelbottom='off', 
                            right='off', 
                            left='off', 
                            labelleft='off')
            idx += 1
            if idx == len(titles):
                break
        if idx == len(titles):
            break
 
    fig.tight_layout()
    return fig
####


def proc_np_hv(pred, return_coords=False):
    """
    Process Nuclei Prediction with XY Coordinate Map

    Args:
        pred:           prediction output, assuming 
                        channel 0 contain probability map of nuclei
                        channel 1 containing the regressed X-map
                        channel 2 containing the regressed Y-map
        return_coords: return coordinates of extracted instances
    """

    blb_raw = pred[...,0]
    h_dir_raw = pred[...,1]
    v_dir_raw = pred[...,2]

    # Processing 
    blb = np.copy(blb_raw)
    blb[blb >= 0.5] = 1
    blb[blb <  0.5] = 0

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1 # background is 0 already
    #####

    h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    h_dir_raw = None  # clear variable
    v_dir_raw = None  # clear variable

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)
    h_dir = None  # clear variable
    v_dir = None  # clear variable

    sobelh = 1 - (cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    sobelv = 1 - (cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

    overall = np.maximum(sobelh, sobelv)
    sobelh = None  # clear variable
    sobelv = None  # clear variable
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    # nuclei values form peaks so inverse to get basins
    dist = -cv2.GaussianBlur(dist,(3, 3),0)

    overall[overall >= 0.5] = 1
    overall[overall <  0.5] = 0
    marker = blb - overall
    overall = None # clear variable
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)
 
    pred_inst = watershed(dist, marker, mask=blb, watershed_line=False)
    if return_coords:
        label_idx = np.unique(pred_inst)
        coords = measurements.center_of_mass(blb, pred_inst, label_idx[1:])
        return pred_inst, coords
    else:
        return pred_inst


def process_instance(pred_map, remap_label=False, output_dtype='uint16'):
    """
    Post processing script for image tiles

    Args:
        pred_map: commbined output of np and hv branches
        remap_label: whether to map instance labels from 1 to N (N = number of nuclei)
        output_dtype: data type of output
    
    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
    """

    pred_inst = np.squeeze(pred_map)
    pred_inst = proc_np_hv(pred_inst)

    # remap label is very slow - only uncomment if necessary to map labels in order
    if remap_label:
        pred_inst = remap_label(pred_inst, by_size=True)

    pred_inst = pred_inst.astype(output_dtype)
    return pred_inst


def process_instance_wsi(pred_map, nr_types, tile_coords, return_masks, remap_label=False, offset=0, output_dtype='uint16'):
    """
    Post processing script

    Args:
        pred_map: commbined output of nc, np and hv branches
        nr_types: number of types considered at output of nc branch
        tile_coords: coordinates of top left corner of tile
        return_masks: whether to save cropped segmentation masks
        remap_label: whether to map instance labels from 1 to N (N = number of nuclei)
        offset: 
        output_dtype: data type of output
    
    Returns:
        mask_list_out: list of cropped predicted segmentation masks
        type_list_out: list of class predictions for each nucleus
        cent_list_out: list of centroid coordinates for each predicted instance (saved as (y,x))
    """

    # init output lists
    mask_list_out = []
    type_list_out = []
    cent_list_out = []

    pred_inst = pred_map[..., nr_types:] # output of combined np and hv branches
    pred_type = pred_map[..., :nr_types] # output of nc branch

    pred_inst = np.squeeze(pred_inst)
    pred_type = np.argmax(pred_type, axis=-1) # pixel wise class mask
    pred_type = np.squeeze(pred_type)

    pred_inst, pred_cent = proc_np_hv(pred_inst, return_coords=True)

    offset_x = tile_coords[0]+offset
    offset_y = tile_coords[1]+offset

    cent_list_out = [(x[0]+offset_y, x[1]+offset_x) for x in pred_cent] # ensure 
    
    # get the shape of the input tile
    shape_pred = pred_inst.shape
    
    # remap label is very slow - only uncomment if necessary to map labels in order
    if remap_label:
        pred_inst = remap_label(pred_inst, by_size=True)

    #### * Get class of each instance id, stored at index id-1
    pred_id_list = list(np.unique(pred_inst))[1:] # exclude background ID
    for idx, inst_id in enumerate(pred_id_list):
        # crop the instance and type masks -> decreases search space and consequently computation time
        crop_inst, crop_type = crop_array(pred_inst, pred_type, pred_cent[idx], shape_pred)
        crop_inst_type = crop_type[crop_inst == inst_id]

        # get the masks cropped at the bounding box
        if return_masks:
            crop_inst_tmp = crop_inst == inst_id
            [rmin, rmax, cmin, cmax] = bounding_box(crop_inst_tmp)
            mask_bbox = crop_inst_tmp[rmin:rmax, cmin:cmax]
            mask_list_out.append(mask_bbox)

        # get the majority class within a given nucleus
        type_list, type_pixels = np.unique(crop_inst_type, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]
        if inst_type == 0: # ! if majority class is background, pick the 2nd most dominant class (if exists)
            if len(type_list) > 1:
                inst_type = type_list[1][0]
        type_list_out.append(inst_type)

    return mask_list_out, type_list_out, cent_list_out
####

def crop_array(pred_inst, pred_type, pred_cent, shape_tile, crop_shape=(70,70)):
    """
    Crop the instance and class array with a given nucleus at the centre.
    Done to decrease the search space and consequently processing time.

    Args:
        pred_inst:  predicted nuclear instances for a given tile
        pred_type:  predicted nuclear types (pixel based) for a given tile
        pred_cent:  predicted centroid for a given nucleus
        shape_tile: shape of tile 
        crop_shape: output crop shape (saved as (y,x))

    Returns:
        crop_pred_inst: cropped pred_inst of shape crop_shape
        crop_pred_type: cropped pred_type of shape crop_shape
    """
    pred_x = pred_cent[1] # x coordinate
    pred_y = pred_cent[0] # y coordinate

    if pred_x < (crop_shape[0]/2):
        x_crop = 0
    elif pred_x > (shape_tile[1] - (crop_shape[1]/2)):
        x_crop = shape_tile[1] - crop_shape[1]
    else:
        x_crop = (pred_cent[1] - (crop_shape[1]/2))
    
    if pred_y < (crop_shape[0]/2):
        y_crop = 0
    elif pred_y > (shape_tile[0] - (crop_shape[0]/2)):
        y_crop = shape_tile[0] - crop_shape[0]
    else:
        y_crop = (pred_cent[0] - (crop_shape[0]/2))
    
    x_crop = int(x_crop)
    y_crop = int(y_crop)
    
    # perform the crop
    crop_pred_inst = pred_inst[y_crop:y_crop+crop_shape[1], x_crop:x_crop+crop_shape[0]]
    crop_pred_type = pred_type[y_crop:y_crop+crop_shape[1], x_crop:x_crop+crop_shape[0]]

    return crop_pred_inst, crop_pred_type
####

def img_min_axis(img):
    """
    Get the minimum of the x and y axes for an input array

    Args:
        img: input array
    """
    try:
        return min(img.shape[:2])
    except AttributeError:
        return min(img.size)
####

def stain_entropy_otsu(img):
    """
    Binarise an input image by calculating the entropy on the 
    hameatoxylin and eosin channels and then using otsu threshold 

    Args:
        img: input array
    """

    img_copy = img.copy()
    hed = skimage.color.rgb2hed(img_copy)  # convert colour space
    hed = (hed * 255).astype(np.uint8)
    h = hed[:, :, 0]
    e = hed[:, :, 1]
    d = hed[:, :, 2]
    selem = disk(4)  # structuring element
    # calculate entropy for each colour channel
    h_entropy = rank.entropy(h, selem)
    e_entropy = rank.entropy(e, selem)
    d_entropy = rank.entropy(d, selem)
    entropy = np.sum([h_entropy, e_entropy], axis=0) - d_entropy
    # otsu threshold
    threshold_global_otsu = threshold_otsu(entropy)
    mask = entropy > threshold_global_otsu

    return mask
####

def morphology(mask, proc_scale):
    """
    Applies a series of morphological operations
    to refine the binarised tissue mask

    Args:
        mask: input binary mask to refine
        proc_scale: scale at which to process
    
    Return:
        processed binary image
    """

    mask_scale = img_min_axis(mask)
    # Join together large groups of small components ('salt')
    radius = int(8 * proc_scale)
    selem = disk(radius)
    mask = binary_dilation(mask, selem)

    # Remove thin structures
    radius = int(16 * proc_scale)
    selem = disk(radius)
    mask = binary_erosion(mask, selem)

    # Remove small disconnected objects
    mask = remove_small_holes(
        mask,
        area_threshold=int(40 * proc_scale)**2,
        connectivity=1,
    )

    # Close up small holes ('pepper')
    mask = binary_closing(mask, selem)

    mask = remove_small_objects(
        mask,
        min_size=int(120 * proc_scale)**2,
        connectivity=1,
    )

    radius = int(16 * proc_scale)
    selem = disk(radius)
    mask = binary_dilation(mask, selem)

    mask = remove_small_holes(
        mask,
        area_threshold=int(40 * proc_scale)**2,
        connectivity=1,
    )

    # Fill holes in mask
    mask = ndimage.binary_fill_holes(mask)

    return mask
####

def get_tissue_mask(img, proc_scale=0.5):
    """
    Obtains tissue mask for a given image

    Args:
        img: input WSI as a np array
        proc_scale: scale at which to process
    
    Returns:
        binarised tissue mask
    """
    img_copy = img.copy()
    if proc_scale != 1.0:
        img_resize = cv2.resize(img_copy, None, fx=proc_scale, fy=proc_scale)
    else:
        img_resize = img_copy

    mask = stain_entropy_otsu(img_resize)
    mask = morphology(mask, proc_scale)
    mask = mask.astype('uint8')

    if proc_scale != 1.0:
        mask = cv2.resize(
            mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask
####

def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger instances has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred # no label
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
#####
