import torch 
import numpy as np 
from mlflow.pytorch import load_model
from collections import deque
import math 
import os 
import glob
import cv2
import argparse

from histocartography.ml.models.hovernet import HoverNet
from histocartography.utils.image import extract_patches_from_image, load_images
from histocartography.utils.hover import rm_n_mkdir, visualize_instances, process_instance


# 1. set constants variables
CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if CUDA else 'cpu'  # either single GPU or cpu 
BATCH_SIZE = 2


def main(args):

    # 1. data loading: single image or from a directory of pngs. 
    all_images = load_images(args.data_path)

    # 2. load HoverNet model from MLflow server 
    if args.model_path:
        model = HoverNet()
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
    else:
        model = load_model('s3://mlflow/6f8ad1831d1846d8bb055ed5ffb24056/artifacts/hovernet_pannuke',  map_location=torch.device('cpu'))
    model.to(DEVICE)

    for (image, image_name) in all_images:

        x = np.array(image) / 255
        im_h = x.shape[0] 
        im_w = x.shape[1]
        sub_patches, nr_step_h, nr_step_w = extract_patches_from_image(image, im_h, im_w)

        pred_map = deque()
        while len(sub_patches) > BATCH_SIZE:
            print('Processing new batch')
            mini_batch  = sub_patches[:BATCH_SIZE]
            sub_patches = sub_patches[BATCH_SIZE:]
            mini_batch = torch.FloatTensor(mini_batch).permute(0,3,1,2).to(DEVICE)
            mini_output = model(mini_batch).cpu().detach().numpy()
            mini_output = np.split(mini_output, BATCH_SIZE, axis=0)
            pred_map.extend(mini_output)
        if len(sub_patches) != 0:
            print('Processing new batch')
            sub_patches = torch.FloatTensor(sub_patches).permute(0,3,1,2).to(DEVICE)
            mini_output = model(sub_patches).cpu().detach().numpy()
            mini_output = np.split(mini_output, len(sub_patches), axis=0)
            pred_map.extend(mini_output)

        output_patch_shape = np.squeeze(pred_map[0]).shape
        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        # Assemble back into full image
        pred_map = np.squeeze(np.array(pred_map))
        pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
        pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
                        np.transpose(pred_map, [0, 2, 1, 3])
        pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1], 
                                            pred_map.shape[2] * pred_map.shape[3], ch))
        pred_map = np.squeeze(pred_map[:im_h,:im_w]) # just crop back to original size

        pred_inst, pred_type = process_instance(pred_map, nr_types=6)
                
        overlaid_output = visualize_instances(image, pred_inst, pred_type)
        overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)

        # combine instance and type arrays for saving
        pred_inst = np.expand_dims(pred_inst, -1)
        pred_type = np.expand_dims(pred_type, -1)
        pred = np.dstack([pred_inst, pred_type])

        cv2.imwrite(os.path.join(args.save_path, image_name), overlaid_output)
        np.save(os.path.join(args.save_path, image_name.replace('png', 'npy')), pred)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        '--save_path',
        type=str,
        help='Save path where the ouput will be stored. Overlaid image + segmentation mask.',
        required=True
    )
    parser.add_argument(
        '--data_path',
        type=str,
        help='Where the images are stored -- works for single image or directories of png.',
        required=True
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='Where the model is saved. If not provided takes the MLflow model.',
        default='',
        required=False
    )

    main(args=parser.parse_args())
