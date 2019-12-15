import glob
import os
import scipy.io as sio
from postproc.hover import proc_np_hv
from config import Config
from utils.utils import *
import json
import time
import csv


class Process(Config):
    def create_directory(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
    # end def

    def run(self):
        # * flag for HoVer-Net
        # 1 - threshold, 2 - sobel based
        energy_mode = 2
        marker_mode = 2
        
        pred_dir = self.inf_output_dir + '_mat/'

        proc_json_dir = self.inf_output_dir + '_json/'
        proc_overlap_dir = self.inf_output_dir + '_overlap/'
        self.create_directory(proc_json_dir)
        self.create_directory(proc_overlap_dir)

        file_list = glob.glob('%s/*.mat' % pred_dir)
        file_list.sort()

        start_time = time.time()
        for filename in file_list:
            filename = os.path.basename(filename)
            basename = filename.split('.')[0]
            print('Working on = ', basename)

            img = cv2.imread(self.inf_data_dir + basename + self.inf_imgs_ext)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
            pred = sio.loadmat('%s/%s.mat' % (pred_dir, basename))
            pred = np.squeeze(pred['result'])
        
            if hasattr(self, 'type_classification') and self.type_classification:
                pred_inst = pred[..., self.nr_types:]
                pred_type = pred[..., :self.nr_types]
        
                pred_inst = np.squeeze(pred_inst)
                pred_type = np.argmax(pred_type, axis=-1)
            else:
                pred_inst = pred

            pred_inst = proc_np_hv(pred_inst, marker_mode=marker_mode, energy_mode=energy_mode, rgb=img)

            pred_inst = remap_label(pred_inst, by_size=True)
            overlaid_output = visualize_instances(pred_inst, img)
            overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('%s/%s.png' % (proc_overlap_dir, basename), overlaid_output)
            pred_inst_features, pred_centroid = extract_feat(img, pred_inst)

            # TODO : remove get_inst_centroid
            pred_inst_centroid = get_inst_centroid(pred_inst)

            j = 0
            for item_t in pred_centroid:
                j += 1
                cv2.drawMarker(overlaid_output, (int(item_t[0]), int(item_t[1])), (0, 255, 0),
                               markerType=cv2.MARKER_STAR,
                               markerSize=10, thickness=1, line_type=cv2.LINE_AA)

                cv2.putText(overlaid_output, str(j), (int(item_t[0]), int(item_t[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 255))  # , 2, cv2.LINE_AA)

            cv2.imwrite('%s/%s.png' % (proc_json_dir, basename), overlaid_output)
        
            # for instance segmentation only
            if self.type_classification:
                pred_id_list = list(np.unique(pred_inst))[1:]  # exclude background ID
                pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
                for idx, inst_id in enumerate(pred_id_list):
                    inst_type = pred_type[pred_inst == inst_id]
                    type_list, type_pixels = np.unique(inst_type, return_counts=True)
                    type_list = list(zip(type_list, type_pixels))
                    type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
                    inst_type = type_list[0][0]
                    if inst_type == 0:  # ! pick the 2nd most dominant if exist
                        if len(type_list) > 1:
                            inst_type = type_list[1][0]
                        else:
                            print('[Warn] Instance has `background` type')
                    pred_inst_type[idx] = inst_type
        
                file_name = '%s/%s.json' % (proc_json_dir, basename)
        
                with open(file_name, 'a') as k:
                    json.dump({'detected_instance_map': pred_inst.tolist(), 'detected_type_map': pred_type.tolist(),
                               'instance_types': pred_inst_type[:, None].tolist(),
                               'instance_centroid_location': pred_centroid.tolist(),
                               'instance_features': pred_inst_features.tolist(),
                               'image_dimension': img.shape}, k)
            else:
        
                file_name = '%s/%s.json' % (proc_json_dir, basename)
                with open(file_name, 'a') as k:
                    json.dump({'detected_instance_map': pred_inst.tolist(),
                               'instance_centroid_location': pred_centroid.tolist(),
                               'instance_features': pred_inst_features.tolist(),
                               'image_dimension': img.shape}, k)

        print('Time per image= ', round((time.time() - start_time)/len(file_list), 2), 's')

    # Save instance centroids to csv, to be used by QuPath
    def save_to_csv(self):
        proc_json_dir = self.inf_output_dir + '_json/'
        proc_csv_dir = self.inf_output_dir + '_csv/'
        self.create_directory(proc_csv_dir)

        file_list = glob.glob('%s/*.json' % proc_json_dir)
        file_list.sort()  # ensure same order

        for filename in file_list:
            basename = os.path.basename(filename).split('.')[0]
            centroid_np = np.zeros(shape=(1, 2))    # x, y

            with open(filename) as f:
                data = json.load(f)
                centroid = data['instance_centroid_location']

                for i in range(len(centroid)):
                    x = centroid[i][0]
                    y = centroid[i][1]
                    centroid_np = np.vstack((centroid_np, np.reshape(np.array([x, y]), newshape=(-1, 2))))

                centroid_np = np.delete(centroid_np, 0, axis=0)
                np.savetxt(proc_csv_dir + basename + '.csv', centroid_np, delimiter=',')
            # end
        # end for
    # end def
# end
















