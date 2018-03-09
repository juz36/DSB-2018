
# coding: utf-8

# In[1]:


import os
import sys
import random
import math
import numpy as np
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
import matplotlib
import matplotlib.pyplot as plt

#import coco
from config import Config
import utils
import model as modellib
import visualize

import torch


# In[2]:


ROOT_DIR = os.getcwd()
#MODEL_DIR = os.path.join(ROOT_DIR, "logs")
TRAIN_PATH = 'data/stage1_train/'
TEST_PATH = 'data/stage1_val/'


# In[3]:


class CellConfig(Config):
    """Configuration for training on data science bowl 2018 dataset.
    Derives from the base Config class and overrides values specific
    to the nuclei images dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cell"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 24, 32, 40)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 670

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    USE_MINI_MASK = False
    
config = CellConfig()
config.display()


# In[46]:


class CellDataset(utils.Dataset):
    def load_cells(self, mode):
        if mode == 'train':
            image_dir = os.path.join(TRAIN_PATH)
        elif mode == 'test':
            image_dir = os.path.join(TEST_PATH)
        image_list = sorted(os.listdir(image_dir))
        
        self.add_class("cells", 1, "cell")
        
        for n, id_ in enumerate(image_list):
            self.add_image(
                "cells",
                image_id=n,
                image_name=id_,
                path=image_dir+'/'+id_+'/images/',
                gt_path=image_dir+'/'+id_+'/masks/'
                )
            
    def load_mask(self, image_id, IMG_HEIGHT=256, IMG_WIDTH=256):
        image_info = self.image_info[image_id]
        #mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        mask = []
        for mask_file in next(os.walk(image_info['gt_path']))[2]:
            mask_ = imread(image_info['gt_path']+mask_file, dtype=np.bool)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                        preserve_range=True), axis=-1)
            mask.append(mask_)
        class_ids = np.ones([len(mask)]).astype(np.int32)
        mask = np.stack(mask,axis=2)
        mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2])
        return mask, class_ids
        
    def load_image(self, image_id, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3):
        image_info = self.image_info[image_id]
        image = imread(image_info['path'] + image_info['image_name'] + '.png')[:,:,:IMG_CHANNELS]
        image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        return image
    


# In[47]:


dataset_train = CellDataset()
dataset_train.load_cells('train')
dataset_train.prepare()
dataset_val = CellDataset()
dataset_val.load_cells('test')
dataset_val.prepare()


# In[50]:


# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     print(mask.shape)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# In[53]:


model = modellib.MaskRCNN(config=config,
                                  model_dir='model/')
if config.GPU_COUNT:
        model = model.cuda()

#model_path = ""

#model.load_weights(model_path)

model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='all')

