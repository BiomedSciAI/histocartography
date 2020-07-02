import xlrd 


loc = "../../data/BRACS-L.xlsx" 
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 


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


for id in range(18, 395):
	split = sheet.cell_value(id, 2)
	if split in WHITE_MARKS:
		troi_name = str(int(sheet.cell_value(id, 0)))
		split = MARKS_TO_SPLIT[sheet.cell_value(id, 2)]
		print(troi_name, split)
		for dataset in AVAILABLE_DATASET:
			# 1. access to the folder with all the images 

			# 2. look to all the ones that are including the TRoI name 

			print('xxx')
	else:
		print('Unrecognized mark')


