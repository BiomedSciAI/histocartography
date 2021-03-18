from PIL import Image 
import numpy as np 
import requests
import zipfile
import io
import os 
import argparse
import glob 

import requests
from tqdm.auto import tqdm
from tqdm import tqdm 

SICAPv2_ZIP_FNAME = 'sicap.zip'
IMAGE_SIZE = 512
STEP = int(IMAGE_SIZE / 2)
IMAGE_REGIONS_DIR = 'image_regions'
MASK_REGIONS_DIR = 'mask_regions'
IMAGE_WSI_DIR = 'image_wsi'
MASK_WSI_DIR = 'mask_wsi'
ID_TO_LABELS = {
    0: 0, 
    3: 1,
    4: 2,
    5: 3,
    255: 4
}

# some regions are duplicates and should not need be included
DUPLICATES = [
    '16B0001851_Block_Region_3',
    '16B0003388_Block_Region_5',
    '16B0003394_Block_Region_1',
    '16B0022608_Block_Region_2',
    '16B0022786_Block_Region_0',
    '16B0023614_Block_Region_3',
    '16B0026792_Block_Region_3',
    '16B0027040_Block_Region_10',
    '18B0005478J_Block_Region_13',
    '18B0005478J_Block_Region_10'
]


def parse_arguments():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help='path to store SICAPv2 for WSIs.',
        required=True
    )
    return parser.parse_args()


def download_zip(url, out_path):
    """"
    Download a zip from a URL

    Args:
        url (str): url to zip file 
        out_path (str): out directory where to dump the zip file
    """
    os.makedirs(out_path, exist_ok=True)
    out_fname = os.path.join(out_path, SICAPv2_ZIP_FNAME)
    if os.path.isfile(out_fname): 
        print('File was already downloaded...')
    else:
        fname = url.split('/')[-1]  
        response = requests.get(url, stream=True)
        with tqdm.wrapattr(open(out_fname, "wb"), "write", miniters=1,
                        total=int(response.headers.get('content-length', 0)),
                        desc=fname) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)


def stitch_image(fnames, slide_id, region_id):
    """
    Stitch a list of patches into a region using patch
    coordinates stored in filenames 

    Args:
        fnames (list): list of patch fnames that belongs to 
                       the same region 
        slide_id (str): name of the WSI 
        region_id (str): name of the region within the WSI 
    
    Returns:
        canvas (PIL.Image): stitched region
    """

    images = [np.array(Image.open(f)) for f in fnames]
    masks = [np.array(Image.open(f.replace('images', 'masks').replace('.jpg', '.png'))) for f in fnames]
    all_x = [int(f.split('/')[-1].split('_')[4]) for f in fnames]
    all_y = [int(f.split('/')[-1].split('_')[5]) for f in fnames]
    image_canvas = np.ones((IMAGE_SIZE + STEP * (max(all_x)), IMAGE_SIZE + STEP * (max(all_y)), 3)) * 255
    mask_canvas = np.ones((IMAGE_SIZE + STEP * (max(all_x)), IMAGE_SIZE + STEP * (max(all_y)))) * 255

    def get_index(my_list, s):
        for idx, e in enumerate(my_list):
            if s in e:
                return idx
        return -1

    for patch_id_x in all_x:
        for patch_id_y in all_y:
            idx = get_index(fnames, slide_id + region_id + '_' + str(patch_id_x) + '_' + str(patch_id_y) + '_')
            if idx != -1:
                image = images[idx]
                mask = masks[idx]

                x_start = patch_id_x * STEP
                x_end = patch_id_x * STEP + IMAGE_SIZE
                y_start = patch_id_y * STEP
                y_end = patch_id_y * STEP + IMAGE_SIZE

                image_canvas[
                    x_start:x_end,
                    y_start:y_end,
                    :
                ] = image

                mask_canvas[
                    x_start:x_end,
                    y_start:y_end
                ] = mask

    def mp(entry, mapper):
        return mapper[entry]

    mask_canvas = np.vectorize(mp)(mask_canvas, ID_TO_LABELS).astype(np.uint8)
    mask_canvas = Image.fromarray(mask_canvas.astype(np.uint8))
    image_canvas = Image.fromarray(image_canvas.astype(np.uint8))
    return image_canvas, mask_canvas


def patches_to_regions(dataset_path):
    """
    Stitch patches into regions (a WSI can be composed of several regions)
    """
    print('Start stitching patches into regions...')

    out_image_path = os.path.join(dataset_path, IMAGE_REGIONS_DIR)
    out_mask_path = os.path.join(dataset_path, MASK_REGIONS_DIR)
    os.makedirs(out_image_path, exist_ok=True)
    os.makedirs(out_mask_path, exist_ok=True)
    image_fnames = glob.glob(os.path.join(dataset_path, 'images', '*.jpg'))
    slide_names = set([f.split('/')[-1].split('_')[0] for f in image_fnames])

    for slide_id in tqdm(slide_names):
        regions_names = set(['_Block_Region_' + f.split('/')[-1].split('_')[3] for f in image_fnames if slide_id in f])
        for region_id in regions_names:
            if (slide_id + region_id) not in DUPLICATES:
                fnames = list(filter(lambda x: slide_id in x and region_id in x, image_fnames))
                image, mask = stitch_image(fnames, slide_id, region_id)
                image.save(os.path.join(out_image_path, slide_id + region_id + '.png'), subsampling=0)
                mask.save(os.path.join(out_mask_path, slide_id + region_id + '.png'), subsampling=0)


def extract_zip(out_path):
    """
    Unzip
    """
    with zipfile.ZipFile(os.path.join(out_path, SICAPv2_ZIP_FNAME), 'r') as zip_ref:
        zip_ref.extractall(out_path)


def regions_to_wsis(dataset_path):
    """
    Stitch regions into WSIs. Most of the WSIs are composed 
    of only one region. 
    """
    print('Start stitching regions into WSIs...')

    out_image_path = os.path.join(dataset_path, IMAGE_WSI_DIR)
    out_mask_path = os.path.join(dataset_path, MASK_WSI_DIR)
    os.makedirs(out_image_path, exist_ok=True)
    os.makedirs(out_mask_path, exist_ok=True)

    mask_fnames = glob.glob(os.path.join(dataset_path, MASK_REGIONS_DIR, '*.png'))
    image_fnames = glob.glob(os.path.join(dataset_path, IMAGE_REGIONS_DIR, '*.png'))

    unique_wsis = set(f.split('/')[-1].split('_')[0] for f in image_fnames)

    for wsi_id in tqdm(unique_wsis):
        region_image_fnames = [r for r in image_fnames if wsi_id in r]
        region_mask_fnames = [r for r in mask_fnames if wsi_id in r]
        if len(region_image_fnames) > 1:

            images = [np.array(Image.open(f)).astype(np.uint8) for f in region_image_fnames]
            masks = [np.array(Image.open(f)).astype(np.uint8) for f in region_mask_fnames]
            width = sum(i.shape[0] for i in images)
            height = max(i.shape[1] for i in images)
            image_canvas = np.ones((width, height, 3)) * 255
            mask_canvas = np.ones((width, height)) * 4  # 4 is background 
            
            x_start = 0
            x_end = images[0].shape[0]
            for idx, (image, mask) in enumerate(zip(images, masks)):
                y_start = 0
                y_end = images[idx].shape[1]
                image_canvas[x_start:x_end, y_start:y_end] = image
                mask_canvas[x_start:x_end, y_start:y_end] = mask
                x_start = x_end
                x_end += images[idx + 1].shape[0] if idx+1 < len(images) else 0

            image_canvas = Image.fromarray(image_canvas.astype(np.uint8))
            mask_canvas = Image.fromarray(mask_canvas.astype(np.uint8))
        else:
            image_canvas = Image.open(region_image_fnames[0])
            mask_canvas = Image.open(region_mask_fnames[0])

        image_canvas.save(os.path.join(out_image_path, wsi_id + '.png'), subsampling=0)
        mask_canvas.save(os.path.join(out_mask_path, wsi_id + '.png'), subsampling=0)
            

if __name__ == "__main__":

    args=parse_arguments()

    # 1. download SICAPv2 dataset 
    download_zip(
        url='https://data.mendeley.com/public-files/datasets/9xxm58dvs3/files/6ab087a7-ca89-47ac-9698-f6546bb50f98/file_downloaded',
        out_path=args.out_path
    )

    # 2. extract zip to destination 
    extract_zip(args.out_path)

    # 3. stitch patches into regions
    patches_to_regions(os.path.join(args.out_path, 'SICAPv2'))

    # 4. stitch regions into WSIs
    regions_to_wsis(os.path.join(args.out_path, 'SICAPv2'))
