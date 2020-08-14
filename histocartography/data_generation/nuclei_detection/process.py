import time
import glob
import scipy.io as sio
from utils import *
from hover import proc_np_hv

class Process:
    def __init__(self, config):
        self.base_image_dir = config.base_image_dir
        self.centroid_output_dir = config.centroid_output_dir
        self.map_output_dir = config.map_output_dir
        self.n_chunks = config.n_chunks
        self.chunk_id = config.chunk_id

        self.inf_imgs_ext = config.inf_imgs_ext
        self.nr_types = config.nr_types
        self.type_classification = config.type_classification

        self.pred_mat_dir = self.map_output_dir + '_mat/'
        #self.proc_overlaid_dir = self.map_output_dir + '_overlaid/'
        self.proc_h5_dir = self.map_output_dir + '_h5/'
        #self.proc_class_dir = self.map_output_dir + '_class/'
        #create_directory(self.proc_overlaid_dir)
        create_directory(self.proc_h5_dir)
        #create_directory(self.proc_class_dir)
    #enddef

    def overlaid_output(self, img, pred_inst, pred_centroid, basename):
        ## Overlay nuclei contours
        overlaid_output = visualize_instances(pred_inst, img)
        overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)

        ## Overlay nuclei markers
        j = 0
        for item_t in pred_centroid:
            j += 1
            cv2.drawMarker(overlaid_output, (int(item_t[0]), int(item_t[1])), (0, 255, 0),
                           markerType=cv2.MARKER_STAR,
                           markerSize=10, thickness=1, line_type=cv2.LINE_AA)

            cv2.putText(overlaid_output, str(j), (int(item_t[0]), int(item_t[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255))  # , 2, cv2.LINE_AA)
        #endfor
        cv2.imwrite('%s/%s.png' % (self.proc_overlaid_dir, basename), overlaid_output)
    #enddef

    def run(self):
        # * flag for HoVer-Net
        # 1 - threshold, 2 - sobel based
        energy_mode = 2
        marker_mode = 2
        
        # ------------------------------------------------------------------------------- CHUNK WISE PROCESSING
        file_list = glob.glob('%s/*%s' % (self.base_image_dir, self.inf_imgs_ext))
        file_list.sort()

        if self.chunk_id != -1 and self.n_chunks != -1:
            idx = np.array_split(np.arange(len(file_list)), self.n_chunks)
            idx = idx[self.chunk_id]
            file_list = [file_list[x] for x in idx]
        print('# Files=', len(file_list))

        # ------------------------------------------------------------------------------- PROCESSING
        total_time = time.time()
        for i in range(len(file_list)):
            start_time = time.time()
            basename = os.path.basename(file_list[i]).split('.')[0]

            if os.path.isfile(self.centroid_output_dir + basename + '.h5') and os.path.isfile(self.proc_h5_dir + basename + '.h5'):
                print('#', i, basename)
                continue

            img = cv2.imread(self.base_image_dir + basename + self.inf_imgs_ext)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # ------------------------------------------------------------------------------- LOADING MAT FILES
            pred = sio.loadmat('%s/%s.mat' % (self.pred_mat_dir, basename))
            pred = np.squeeze(pred['result'])
        
            if hasattr(self, 'type_classification') and self.type_classification:
                pred_inst = pred[..., self.nr_types:]
                pred_type = pred[..., :self.nr_types]
        
                pred_inst = np.squeeze(pred_inst)
                pred_type = np.argmax(pred_type, axis=-1)
            else:
                pred_inst = pred
            #endif

            # ------------------------------------------------------------------------------- POST-PROCESSING
            pred_inst = proc_np_hv(pred_inst, marker_mode=marker_mode, energy_mode=energy_mode, rgb=img)
            pred_inst = remap_label(pred_inst, by_size=False)      # by_size=False: Saves time

            # ------------------------------------------------------------------------------- CENTROID EXTRACTION
            pred_centroid = extract_centroid(pred_inst)

            if len(pred_centroid) == 0:
                print('ERROR: M["m00"] = 0 ', basename)

                regions = regionprops(pred_inst)
                pred_centroid = np.zeros(shape=(len(regions), 2))
                for i, region in enumerate(regions):
                    pred_centroid[i, :] = region['centroid']  # (row, column) = (y, x)
                pred_centroid.T[[0, 1]] = pred_centroid.T[[1, 0]]

            # ------------------------------------------------------------------------------- FEATURE EXTRACTION
            pred_inst_features = extract_feat(img, pred_inst)

            #------------------------------------------------------------------------------- OVERLAID OUTPUT PLOT
            # Comment out (this block + overlaid_output lines from above) to save time.
            #self.overlaid_output(img, pred_inst, pred_centroid, basename)

            #------------------------------------------------------------------------------- SAVE PREDICTIONS AS JSON & H5
            # For instance segmentation only
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
                #endfor

                save_centroid_h5(self.centroid_output_dir + basename + '.h5', pred_centroid, img.shape)
                save_instance_map_h5(self.proc_h5_dir + basename + '.h5', pred_inst)
                save_class_type_h5(self.proc_class_dir + basename + '.h5', pred_type, pred_inst_type[:, None])
        
            else:
                save_centroid_h5(self.centroid_output_dir + basename + '.h5', pred_centroid, img.shape)
                save_instance_map_h5(self.proc_h5_dir + basename + '.h5', pred_inst)
            #endif

            print('#', i,  ' working on = ', basename, ' #nuclei = ', pred_inst_features.shape[0], ' time = ', round(time.time() - start_time, 2))
        #endfor

        print('Time per image= ', round((time.time() - total_time)/len(file_list), 2), 's')
    #enddef

#end
















