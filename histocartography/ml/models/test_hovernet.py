import torch 
import tensorflow as tf 
import six 
import numpy as np 
from tensorpack.tfutils.common import get_op_tensor_name, get_tf_version_tuple
import mlflow.pytorch

from histocartography.ml.models.hovernet import HoverNet

# Pre-load weights in dict 
weights = np.load('hovernet.npz')
dic = {get_op_tensor_name(k)[1]: v for k, v in six.iteritems(weights)}
var_to_dump = set(dic.keys())
dic_to_dump = {k: v for k, v in six.iteritems(dic) if k in var_to_dump}
dic_to_dump = dict(sorted(dic_to_dump.items()))

tf_dict = {}
for key, val in dic_to_dump.items():
    if 'mean' not in key and 'variance' not in key:
        # key = key.replace(':0', '')
        key = key.replace('/W:0', '.weight')
        key = key.replace('/bn/beta:0', '.bn.bias')
        key = key.replace('/bn/gamma:0', '.bn.weight')
        key = key.replace('/b:0', '.bias')
        key = key.replace('/', '.')
        key = key.replace('_', '.')
        tf_dict[key] = val

# declare model 
model = HoverNet()

# get a dummy image 
# x = torch.randn(1, 3, 256, 256)
# print('Input shape:', x.shape)

# out = model(x)

pt_keys = []
for name, param in model.named_parameters():
    pt_keys.append(name)

def preprocess_pt_key(name):
    name = name.replace('_', '.')
    name = name.replace('.conv.weight', '.weight')
    name = name.replace('.conv.bias', '.bias')
    name = name.replace('.act.', '.')
    name = name.replace('.', '')
    return name

def find_substr(query, all_strings):
    query_processed = query.replace('.', '')
    found = None
    for string in all_strings:
        string_processed = preprocess_pt_key(string)
        if query_processed in string_processed:
            found = string
    return found

for tf_key, tf_val in tf_dict.items():
    out = find_substr(tf_key, pt_keys)
    if out is not None:
        print('Found:', tf_key, 'in pytorch key:', out)
        pt_keys.remove(out)
        # set manually the weights into the pytorch model
        exec('model.' + out + '.data' + ' = ' + 'torch.FloatTensor(tf_val)')
    else:
        print('Could not find:', tf_key)

mlflow.pytorch.log_model(model, 'hovernet_pannuke')