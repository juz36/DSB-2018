"""

"""
import os
import numpy as np
from utils.path import data_dir, train_dir
from dataset.dataset import PrepareDataset
from matplotlib.pyplot import imsave


# mkdir if not exists
if not os.path.exists(train_dir + '/images'):
    os.makedirs(train_dir + '/images')
if not os.path.exists(train_dir + '/multi_masks'):
    os.makedirs(train_dir + '/multi_masks')

dataset = PrepareDataset(data_dir+'/stage1_train')

# preprocess training images & generate masks
for img_id, image, masks in dataset:
    imsave(train_dir + '/images/%s.png' % img_id, image)
    imsave(train_dir + '/images/%s_mask.png' % img_id, masks)
    np.save(train_dir + '/multi_masks/%s.npy' % img_id, masks)
