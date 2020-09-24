import torch 
import numpy as np 
from mlflow.pytorch import load_model
from collections import deque
import math 

from histocartography.ml.models.hovernet import HoverNet


# 2. load image (dummy one for now)
x = torch.randn(1, 3, 256*3, 256*3)

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

# 1. load HoverNet model from MLflow server 
model = load_model('s3://mlflow/5b0b548adfdc4214927478e95311d30b/artifacts/hovernet_pannuke',  map_location=torch.device('cpu'))

pred_map = deque()
while len(sub_patches) > batch_size:
    mini_batch  = sub_patches[:batch_size]
    sub_patches = sub_patches[batch_size:]
    mini_output = model(mini_batch)
    mini_output = np.split(mini_output, batch_size, axis=0)
    pred_map.extend(mini_output)
if len(sub_patches) != 0:
    mini_output = model(sub_patches)
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
