import time
import glob
import math
import importlib
from scipy import io as sio
from collections import deque
from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils.sessinit import get_model_loader
from utils import *

class Infer:
    def __init__(self, config):
        self.base_image_dir = config.base_image_dir
        self.inf_model_path = config.inf_model_path
        self.centroid_output_dir = config.centroid_output_dir
        self.map_output_dir = config.map_output_dir
        self.n_chunks = config.n_chunks
        self.chunk_id = config.chunk_id

        self.infer_mask_shape = config.infer_mask_shape
        self.infer_input_shape = config.infer_input_shape
        self.inf_batch_size = config.inf_batch_size
        self.inf_imgs_ext = config.inf_imgs_ext
        self.eval_inf_input_tensor_names = config.eval_inf_input_tensor_names
        self.eval_inf_output_tensor_names = config.eval_inf_output_tensor_names
    #enddef

    def get_model(self):
        model_constructor = importlib.import_module('model.hovernet_nuclei_segmentation')
        model_constructor = model_constructor.Model_NP_HV
        return model_constructor
    #enddef

    def gen_prediction(self, x, predictor):
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
        pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else np.transpose(pred_map, [0, 2, 1, 3])
        pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1], pred_map.shape[2] * pred_map.shape[3], ch))
        pred_map = np.squeeze(pred_map[:im_h, :im_w])
        return pred_map
    #enddef

    def run(self, config):
        create_directory(self.map_output_dir + '_mat/')

        print("Loaded model weight file")
        model_constructor = self.get_model()
        pred_config = PredictConfig(model=model_constructor(config),
                                    session_init=get_model_loader(self.inf_model_path),
                                    input_names=self.eval_inf_input_tensor_names,
                                    output_names=self.eval_inf_output_tensor_names)
        predictor = OfflinePredictor(pred_config)

        # ------------------------------------------------------------------------------- CHUNK WISE PROCESSING
        file_list = glob.glob('%s/*%s' % (self.base_image_dir, self.inf_imgs_ext))
        file_list.sort()

        if self.chunk_id != -1 and self.n_chunks != -1:
            idx = np.array_split(np.arange(len(file_list)), self.n_chunks)
            idx = idx[self.chunk_id]
            file_list = [file_list[x] for x in idx]
        print('# Files=', len(file_list))

        # ------------------------------------------------------------------------------- INFER
        total_time = time.time()
        for i in range(len(file_list)):
            start_time = time.time()
            basename = os.path.basename(file_list[i]).split('.')[0]

            img = cv2.imread(file_list[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred_map = self.gen_prediction(img, predictor)
            sio.savemat('%s/%s.mat' % (self.map_output_dir + '_mat/', basename), {'result': [pred_map]})
            print('#' , i, ' working on = ', basename, ' time=', round(time.time() - start_time, 2))
        #endfor

        print('Time per image= ', round((time.time() - total_time)/len(file_list), 2), 's')
    #endfor
#end