import pandas as pd
import numpy as np
import torch
import scipy
import h5py
import zipfile
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from itertools import permutations

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


import numpy as np
from PIL import Image
import random

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        # breakpoint()
        image= sample.data['images']
        depth = sample.data['depths']

        if not isinstance(image, np.ndarray):
            raise TypeError(
                'img should be numpy array. Got {}'.format(type(image)))
        
        for i in range(image.shape[0]):
            if random.random() < 0.5:
                image[i] = np.fliplr(image[i])

        return {'image': image, 'depth': depth}

# class RandomChannelSwap(object):
#     def __init__(self, probability):
#         self.probability = probability
#         self.indices = list(permutations(range(3), 3))

#     def __call__(self, sample):
#         image, depth = sample['image'], sample['depth']
#         if not isinstance(image, np.ndarray): 
#             raise TypeError('img should be numpy array. Got {}'.format(type(image)))
#         if random.random() < self.probability:
#             permuted_indices = self.indices[random.randint(0, len(self.indices) - 1)]
#             image = image[..., permuted_indices]
#         return {'image': image, 'depth': depth}




class depthDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __getitem__(self, idx):
        # breakpoint()
        image = self.data['images'][idx]
        depth = self.data['depths'][idx]
        sample = {'image': image, 'depth': depth}
        if self.transform: sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.data['images'].shape[0]

class ToTensor(object):
    def __init__(self,is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        # breakpoint()
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image, False)

        if self.is_test:
            depth = self.to_tensor(depth, True).float() / 1000
        else:            
            depth= self.to_tensor(depth, True).float() * 1000
            
        depth = torch.clamp(depth, 10, 1000)
        
        # put in expected range

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic, isdepth):
        if isdepth:
            depth = torch.from_numpy(pic)
            return depth.float().div(255)
        else:
            img = torch.from_numpy(pic)
            return img.float().div(255)
            # new_pic = np.empty((pic.shape[0], pic.shape[3], pic.shape[1], pic.shape[2]), dtype=np.uint8)
            # new_pic = torch.from_numpy(pic.transpose((0, 3, 1, 2)))
            # return new_pic.float().div(255)

def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])

def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        ToTensor()
    ])


def loadMatToMem(zip_file):
    # Load mat file into memory
    print('Loading nyu dataset...', end='')
    # unpack the h5py dataset
    # keys inside: 
    # ['#refs#', '#subsystem#', 'accelData', 'depths', 'images', 'instances', 'labels', 'names', 'namesToIds', 'rawDepthFilenames', 'rawDepths', 'rawRgbFilenames', 'sceneTypes', 'scenes']
    with h5py.File(zip_file, 'r') as file:
        # breakpoint()
        data = {}
        # for key in file.keys():
        #     data[key] = file[key][()]
        # breakpoint()
        data['images'] = file['images'][()]
        data['depths'] = file['depths'][()]
        return data

    # TODO: shuffle the data set 
    # from sklearn.utils import shuffle
    # nyu2_train = shuffle(nyu2_train, random_state=0)





def getTrainingTestingData(batch_size):

    data = loadMatToMem('nyu_depth_v2_labeled.mat')

    # Assuming you have your data stored in a dictionary named 'data'
    # with keys 'images' and 'depths'
    images = data['images']
    depths = data['depths']

    # Specify the number of folds
    num_folds = 5

    # Initialize the KFold object
    kf = KFold(n_splits=num_folds, shuffle=True)

    # Iterate over the splits
    for fold, (train_idx, test_idx) in enumerate(kf.split(images)):
        # print(f'Fold {fold+1}:')
        # Split the data into train and test sets
        train_data = {'images': images[train_idx], 'depths': depths[train_idx]}
        test_data = {'images': images[test_idx], 'depths': depths[test_idx]}  

    transformed_training = depthDataset(train_data, transform=getNoTransform())
    transformed_testing = depthDataset(test_data, transform=getNoTransform())
    # breakpoint()
    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size, shuffle=False)

    


