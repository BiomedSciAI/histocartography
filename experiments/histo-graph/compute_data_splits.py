import xlrd 
import os 
from glob import glob


BASE_PATH = '/dataT/pus/histocartography/Data/BRACS_L'
IMAGE_PATH = os.path.join(BASE_PATH, 'Images_norm')
OUT_PATH = os.path.join(BASE_PATH, 'data_split_cv', 'data_split_1')
AVAILABLE_DATASET = ['benign', 'pathologicalbenign', 'udh', 'adh', 'fea', 'dcis', 'malignant']
AVAILABLE_SPLITS = ['train', 'test', 'val']


for split in AVAILABLE_SPLITS:
	for dataset in AVAILABLE_DATASET:
		name = split + '_list_' + dataset
		vars()[name] = []


CELL_ID_TO_DATASET = {
	3: 'benign',
	4: 'pathologicalbenign',
	5: 'udh',
	6: 'adh',
	7: 'fea',
	8: 'dcis',
	9: 'malignant'
}


MARKS_TO_SPLIT = {
	'XXX': 'train',
	'XX': 'val', 
	'X': 'test',
	'': None
}


WHITE_MARKS = ['XXX', 'XX', 'X']

# Core starts here
wb = xlrd.open_workbook(os.path.join(BASE_PATH, 'BRACS-L.xlsx')) 
sheet = wb.sheet_by_index(0) 

def is_valid(troi_name, wsi_name):
	return troi_name.startswith(wsi_name)


for id in range(18, 395):
	split = sheet.cell_value(id, 2)
	if split in WHITE_MARKS:
		wsi_name = str(int(sheet.cell_value(id, 0)))
		split = MARKS_TO_SPLIT[sheet.cell_value(id, 2)]
		for dataset in AVAILABLE_DATASET:
			# 1. access to the folder with all the images 
			dir_name = os.path.join(IMAGE_PATH, dataset)

			# 2. look to all the ones that are including the TRoI name 
			all_images = glob(os.path.join(dir_name, '*.png'))
			sub_images = [image.split('/')[-1].replace('.png', '') for image in all_images if is_valid(image.split('/')[-1], wsi_name)]

			# 3. append the images
			exec(split + '_list_' + dataset + '.extend(sub_images)')


# write as txt files
for split in AVAILABLE_SPLITS:
	for dataset in AVAILABLE_DATASET:
		var_name = split + '_list_' + dataset
		out_fname = split + '_list_' + dataset + '.txt'
		exec('print(' + var_name + ')')
		with open(os.path.join(OUT_PATH, out_fname), 'w') as filehandle:
		    filehandle.writelines("%s\n" % place for place in vars()[var_name])

		
