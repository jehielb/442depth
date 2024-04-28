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


def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}

class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}



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
        image, depth = sample['image'], sample['depth']
        
        image = self.to_tensor(image, False)

        # depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth, True).float() / 1000
        else:            
            depth = self.to_tensor(depth, True).float() * 1000
        
        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic, isdepth):
        # if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        #     raise TypeError(
        #         'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        # breakpoint()
        if isdepth:
            depth = torch.from_numpy(pic)
            return depth.float().div(255)
        else:
            img = torch.from_numpy(pic.transpose((0, 1, 2)))
            return img.float().div(255)

def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])

def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])


def loadMatToMem(zip_file):
    # Load mat file into memory
    print('Loading nyu dataset...', end='')
    # unpack the h5py dataset
    # keys inside: 
    # ['#refs#', '#subsystem#', 'accelData', 'depths', 'images', 'instances', 'labels', 'names', 'namesToIds', 'rawDepthFilenames', 'rawDepths', 'rawRgbFilenames', 'sceneTypes', 'scenes']
    with h5py.File('nyu_depth_v2_labeled.mat', 'r') as file:
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

    data = loadMatToMem('nyu_data.zip')
    breakpoint()

    output_np = (output_np * 255).astype(np.uint8)
    # Convert NumPy array to PIL image
    output_img = Image.fromarray(output_np)
    # Save the image to a file
    output_img.save('output_image.png')
    
    transformed_training = depthDataset(data, transform=getDefaultTrainTransform())
    transformed_testing = depthDataset(data, transform=getNoTransform())
    # breakpoint()
    # test1 = DataLoader(transformed_training, batch_size, shuffle=True)
    # test2 = DataLoader(transformed_testing, batch_size, shuffle=False)
    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size, shuffle=False)

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
        print(f'Fold {fold+1}:')
        # Split the data into train and test sets
        train_images, train_depths = images[train_idx], depths[train_idx]
        test_images, test_depths = images[test_idx], depths[test_idx]


def colorize_depth(image_file, colormap='viridis'):
    # Normalize depth values to [0, 1]
    image = Image.open(image_file)

    # Convert image to numpy array
    depth_map = np.array(image)

    normalized_depth = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

    # Choose colormap
    cmap = plt.get_cmap(colormap) # matplotlib.pyplot

    # Map normalized depth values to colors
    colored_depth_map = cmap(normalized_depth)

    # Convert to uint8 RGB image
    colored_depth_map_rgb = (colored_depth_map[:, :, :3] * 255).astype(np.uint8)

    output_img = Image.fromarray(colored_depth_map_rgb)
    # Save the image to a file
    output_img.save('depth_map.png')

    return colored_depth_map_rgb

