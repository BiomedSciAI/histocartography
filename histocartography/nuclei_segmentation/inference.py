import glob
import math
import os
from collections import deque
from process import Process
import cv2
import numpy as np
from scipy import io as sio
from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils.sessinit import get_model_loader
from config import Config
import time


from absl import logging
logging._warn_preinit_stderr = 0

import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('tensorpack').disabled = True

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


class Inferer(Config):
    def __gen_prediction(self, x, predictor):
        """
        Using 'predictor' to generate the prediction of image 'x'

        Args:
            x : input image to be segmented. It will be split into patches
                to run the prediction upon before being assembled back
            predictor: using HoverNet Model to produce the predicated instance map for the image x

        Result:
            Returns the predcted instance map(instance map and type classification if classification=True)
        """
        step_size = self.infer_mask_shape
        msk_size = self.infer_mask_shape
        win_size = self.infer_input_shape

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        im_h = x.shape[0]
        im_w = x.shape[1]

        last_h, nr_step_h = get_last_steps(im_h, msk_size[0], step_size[0])
        last_w, nr_step_w = get_last_steps(im_w, msk_size[1], step_size[1])

        diff_h = win_size[0] - step_size[0]
        padt = diff_h // 2
        padb = last_h + win_size[0] - im_h

        diff_w = win_size[1] - step_size[1]
        padl = diff_w // 2
        padr = last_w + win_size[1] - im_w

        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), 'reflect')

        sub_patches = []
        for row in range(0, last_h, step_size[0]):
            for col in range(0, last_w, step_size[1]):
                win = x[row:row + win_size[0], col:col + win_size[1]]
                sub_patches.append(win)

        pred_map = deque()
        while len(sub_patches) > self.inf_batch_size:
            mini_batch = sub_patches[:self.inf_batch_size]
            sub_patches = sub_patches[self.inf_batch_size:]
            mini_output = predictor(mini_batch)[0]
            mini_output = np.split(mini_output, self.inf_batch_size, axis=0)
            pred_map.extend(mini_output)
        if len(sub_patches) != 0:
            mini_output = predictor(sub_patches)[0]
            mini_output = np.split(mini_output, len(sub_patches), axis=0)
            pred_map.extend(mini_output)

        output_patch_shape = np.squeeze(pred_map[0]).shape
        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        pred_map = np.squeeze(np.array(pred_map))
        pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
        pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
            np.transpose(pred_map, [0, 2, 1, 3])
        pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1],
                                         pred_map.shape[2] * pred_map.shape[3], ch))
        pred_map = np.squeeze(pred_map[:im_h, :im_w])
        return pred_map

    def create_directory(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def run(self):
        print("Loaded model weight file")
        model_path = self.inf_model_path

        model_constructor = self.get_model()
        pred_config = PredictConfig(
            model=model_constructor(),
            session_init=get_model_loader(model_path),
            input_names=self.eval_inf_input_tensor_names,
            output_names=self.eval_inf_output_tensor_names)
        predictor = OfflinePredictor(pred_config)

        save_dir = self.inf_output_dir
        self.create_directory(save_dir)
        save_dir += '_mat/'
        self.create_directory(save_dir)

        file_list = glob.glob('%s/*%s' % (self.inf_data_dir, self.inf_imgs_ext))
        file_list.sort()

        start_time = time.time()
        for filename in file_list:
            filename = os.path.basename(filename)
            basename = filename.split('.')[0]
            print('Working on = ', basename)

            # print("Reading images")
            img = cv2.imread(self.inf_data_dir + filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # print("Generating prediction map")
            pred_map = self.__gen_prediction(img, predictor)
            sio.savemat('%s/%s.mat' % (save_dir, basename), {'result': [pred_map]})

        print('Time per image= ', round((time.time() - start_time)/len(file_list), 2), 's')

# end


if __name__ == '__main__':
    print('Inferencing...')
    inferer = Inferer()
    inferer.run()

    print('\nProcessing...')
    process = Process()
    process.run()
    process.save_to_csv()

