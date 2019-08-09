"""Tumor Slide Classification ."""
import logging
import sys
import tensorflow as tf
import numpy as np


# setup logging
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::ML::TumorSlidePrediction')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
h1.setFormatter(formatter)
log.addHandler(h1)


def predict_for_image(patch_info_coordinates, image=None, model_json=None, model_weights=None):
    all_patches = []
    for i, loc in enumerate(patch_info_coordinates):
        patch_image = image[loc[1]: loc[2], loc[3]:loc[4], :]
        all_patches.append(patch_image)

    all_patches = np.array(all_patches)
    print('all patches shape: ', all_patches.shape)
    return len(all_patches)

    '''
    batch_size = 32
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights)
    print('Model loaded')

    datagen = ImageDataGenerator(rescale=1. / 255)
    datagen.fit(all_patches)
    n_steps = np.ceil(len(all_patches) / batch_size)
    y_pred = model.predict_generator(datagen.flow(all_patches, batch_size=batch_size), steps=n_steps)
    print('time for prediction:', (start6 - start5))
    print('y_pred : ', len(y_pred))
    print('y_pred : ', y_pred[0:10])
    #'''




