import os
import glob
import copy
from shutil import copy2

base_src_path = '/dataT/pus/histocartography/Data/BRACS_L/'
base_dst_path = '/dataT/pus/histocartography/Data/explainability/'
tumor_types = ['benign', 'pathologicalbenign', 'udh', 'adh', 'fea', 'dcis', 'malignant']

def create_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)

types = ['nuclei_info/nuclei_detected/centroids/',
         'nuclei_info/nuclei_detected/instance_map/',
         'nuclei_info/nuclei_features/features_cnn_resnet34_mask_False_/',
         'nuclei_info/nuclei_features/features_cnn_resnet50_mask_False_/',

         'super_pixel_info/sp_merged_detected/merging_hc/centroids/',
         'super_pixel_info/sp_merged_detected/merging_hc/instance_map/',
         'super_pixel_info/sp_merged_features/merging_hc_features_cnn_resnet34_mask_False_/',
         'super_pixel_info/sp_merged_features/merging_hc_features_cnn_resnet50_mask_False_/',
         'super_pixel_info/sp_unmerged_detected/centroids/',
         'super_pixel_info/sp_unmerged_detected/instance_map/',
         'super_pixel_info/sp_unmerged_features/features_cnn_resnet34_mask_False_/',
         'super_pixel_info/sp_unmerged_features/features_cnn_resnet50_mask_False_/',

         'graphs/cell_graph_model/features_cnn_resnet34_mask_False_/',
         'graphs/cell_graphs/features_cnn_resnet34_mask_False_/',
         'graphs/cell_graphs/features_cnn_resnet50_mask_False_/',

         'graphs/superpx_graph_model/merging_hc_features_cnn_resnet34_mask_False_/',
         'graphs/tissue_graphs/merging_hc_features_cnn_resnet34_mask_False_/',
         'graphs/tissue_graphs/merging_hc_features_cnn_resnet50_mask_False_/',
         'assignment_mat/'
         ]
extensions = ['.h5', '.h5',  '.h5', '.h5',
              '.h5', '.h5', '.h5', '.h5', '.h5', '.h5', '.h5', '.h5',
              '.bin', '.bin', '.bin',
              '.bin', '.bin', '.bin', '.npy']

# Get file names
base_img_path = base_dst_path + 'Images_norm/'
filenames = []
for t in tumor_types:
    paths = glob.glob(base_img_path + t + '/*.png')
    paths.sort()
    filenames += paths

filenames = [os.path.basename(x).split('.')[0] for x in filenames]

# Start copying
for i in range(len(types)):
    if i != 15:
        continue

    count = 0
    dirs = types[i].split('/')

    path = copy.deepcopy(base_dst_path)
    for j in range(len(dirs)):
        path += dirs[j] + '/'
        create_directory(path)

    src_path = base_src_path + types[i]
    dst_path = base_dst_path + types[i]

    for t in tumor_types:
        create_directory(dst_path + t)

    for j in range(len(filenames)):
        tumor_type = filenames[j].split('_')[1]

        if 'nuclei_info/nuclei_detected/instance_map/' == types[i]:
            create_directory(src_path + tumor_type + '/_h5/')
            create_directory(dst_path + tumor_type + '/_h5/')

            src = src_path + tumor_type + '/_h5/' + filenames[j] + extensions[i]
            dst = dst_path + tumor_type + '/_h5/'
        else:
            src = src_path + tumor_type + '/' + filenames[j] + extensions[i]
            dst = dst_path + tumor_type + '/'

        if os.path.isfile(src):
            copy2(src, dst)
            count += 1

    print(types[i], ' #Copied files=', count)













