import torch 
import numpy as np 
# from mlflow.pytorch import load_model
from collections import deque
import math 
import cv2

from histocartography.ml.models.hovernet import HoverNet

from hover.misc.utils import rm_n_mkdir
from hover.misc.viz_utils import visualize_instances
import hover.postproc.process_utils as proc_utils

# 1. cuda available
cuda = torch.cuda.is_available()

# 2. test with an image 
image = cv2.imread("/Users/gja/Documents/PhD/histocartography/data/Scan6_7_8_9_10/Images_normv2/0_benign/1937_benign_4.png")  # hardcoded path 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
x = np.array(image) / 255

# 3. patch-based processing of the input 
batch_size = 2
step_size = [164, 164]
msk_size = [164, 164]
win_size = [256, 256]

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

x = np.pad(x, ((padt, padb), (padl, padr), (0, 0)), 'reflect')

sub_patches = []
# generating subpatches from orginal
for row in range(0, last_h, step_size[0]):
    for col in range (0, last_w, step_size[1]):
        win = x[row:row+win_size[0], 
                col:col+win_size[1]]
        sub_patches.append(win)
print('Numberof patches:', len(sub_patches))

# 1. load HoverNet model from MLflow server 
# model = load_model('s3://mlflow/5b0b548adfdc4214927478e95311d30b/artifacts/hovernet_pannuke',  map_location=torch.device('cpu'))
model = HoverNet()
model.load_state_dict(torch.load('hovernet.pt'))
model.eval()
if cuda:
    model = model.cuda()

pred_map = deque()
while len(sub_patches) > batch_size:
    mini_batch  = sub_patches[:batch_size]
    sub_patches = sub_patches[batch_size:]
    mini_batch = torch.FloatTensor(mini_batch).permute(0,3,1,2)
    if cuda:
        mini_batch = mini_batch.cuda()
    print('Input dimensions are:', mini_batch.shape)
    mini_output = model(mini_batch).cpu().detach().numpy()
    mini_output = np.split(mini_output, batch_size, axis=0)
    pred_map.extend(mini_output)
if len(sub_patches) != 0:
    sub_patches = torch.FloatTensor(sub_patches).permute(0,3,1,2)
    if cuda:
        sub_patches = sub_patches.cuda()
    print('Input dimensions are:', sub_patches.shape)
    mini_output = model(sub_patches).cpu().detach().numpy()
    mini_output = np.split(mini_output, len(sub_patches), axis=0)
    pred_map.extend(mini_output)

output_patch_shape = np.squeeze(pred_map[0]).shape
ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

# Assemble back into full image
pred_map = np.squeeze(np.array(pred_map))
pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
                np.transpose(pred_map, [0, 2, 1, 3])
pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1], 
                                    pred_map.shape[2] * pred_map.shape[3], ch))
pred_map = np.squeeze(pred_map[:im_h,:im_w]) # just crop back to original size

pred_inst, pred_type = proc_utils.process_instance(pred_map, nr_types=6)
         
overlaid_output = visualize_instances(image, pred_inst, pred_type)
overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)

# combine instance and type arrays for saving
pred_inst = np.expand_dims(pred_inst, -1)
pred_type = np.expand_dims(pred_type, -1)
pred = np.dstack([pred_inst, pred_type])

cv2.imwrite('output.png', overlaid_output)
