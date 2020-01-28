import time
import glob
import os
from utils_sp import *

tumor_type = '0_benign'       # 0_benign, 1_pathological_benign, 2_udh, 3_adh, 4_fea, 5_dcis, 6_malignant

inf_data_dir = '/dataT/pus/histocartography/Data/PASCALE/Images/'
inf_output_dir = '/dataT/pus/histocartography/Data/PASCALE/nuclei_info/Predictions/' + tumor_type + '/'
proc_h5_dir = inf_output_dir + '_h5/'
proc_overlap_dir = inf_output_dir + '_nuclei_loc/'
inf_imgs_ext = '.png'

if not os.path.isdir(proc_overlap_dir):
    os.mkdir(proc_overlap_dir)

#------------------------------------------------------------------------------- MAIN CODE
file_list = glob.glob(proc_h5_dir + '*.h5')
file_list.sort()
nuclei_counter = []

for i in range(len(file_list)):
    start_time = time.time()
    filename = os.path.basename(file_list[i])
    basename = filename.split('.')[0]

    img = cv2.imread(inf_data_dir + tumor_type + '/' + basename + inf_imgs_ext)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ## Read h5 file
    f = h5py.File(file_list[i], 'r')
    pred_centroid = f['instance_centroid_location']

    #------------------------------------------------------------------------------- OVERLAID OUTPUT PLOT
    if pred_centroid.shape[0] != 0:
        for k in range(pred_centroid.shape[0]):
            x = pred_centroid[k, 0]
            y = pred_centroid[k, 1]
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        #endfor
    #endif
    overlaid_output = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite('%s/%s.png' % (proc_overlap_dir, basename), overlaid_output)
    nuclei_counter.append(pred_centroid.shape[0])

    print('#', i, ' working on = ', basename, ' #nuclei = ', pred_centroid.shape[0], ' time = ', round(time.time() - start_time, 2))
#endfor

nuclei_counter = np.asarray(nuclei_counter)

print('Nuclei summary: min= ', np.min(nuclei_counter), ' max= ', np.max(nuclei_counter), ' avg= ', int(np.mean(nuclei_counter)))
