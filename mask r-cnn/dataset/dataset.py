"""
Dataset & DataLoader for DSB2018

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import rgba2rgb
from skimage.io import imread
from torch.utils.data import Dataset

from dataset.rle import read_train_rles, rle_decode_location
from utils.path import data_dir


class PrepareDataset(Dataset):

    def __init__(self, img_path, rles_path=data_dir+'/stage1_train_labels.csv', transform=None):
        # initialize
        super(PrepareDataset, self).__init__()
        self.__transform = transform
        self.__img_path = img_path

        # load image_id and RLEs from csv
        rles_dict = read_train_rles(rles_path)
        self.__rles_dict = rles_dict
        img_ids = sorted(rles_dict.keys())
        self.__img_ids = img_ids

    def __len__(self):
        return len(self.__img_ids)

    def __getitem__(self, index):
        img_id = self.__img_ids[index]

        # load image
        image = imread(self.__img_path+'/%s/images/%s.png' % (img_id, img_id))
        if image.shape[2] != 3:
            image = rgba2rgb(image)

        # load mask
        rles = self.__rles_dict[img_id]
        multi_mask = np.zeros(np.prod(image.shape[0:2]), np.uint8)
        for i, rle in enumerate(rles):
            mask_location = rle_decode_location(rle)
            for low, high in mask_location:
                multi_mask[low:high] = i+1
        multi_mask = multi_mask.reshape(image.shape[1::-1]).T
        return img_id, image, multi_mask


class CellDataset(Dataset):

    def __init__(self, img_path, rles_path=data_dir+'/stage1_train_labels.csv', transform=None, mode='train'):

        # initialize
        assert mode == 'train' or mode == 'test'
        super(CellDataset, self).__init__()
        self.__transform = transform
        self.__mode = mode
        self.__img_path = img_path

        # load image_id from csv
        train_labels = pd.read_csv(rles_path)
        img_ids = sorted(train_labels["ImageId"])
        self.__img_ids = img_ids

    def __len__(self):
        return len(self.__img_ids)

    def __getitem__(self, index):
        img_id = self.__img_ids[index]

        # load image
        image = imread(self.__img_path+'/images/%s.png' % img_id)

        # <todo> transformations

        if self.__mode == 'train':
            # load mask
            multi_mask = np.load(self.__img_path + '/%s.npy' % img_id)
            return image, multi_mask
        else:
            return image
