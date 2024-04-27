import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import tensorflow as tf

class UpProject(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1):
        """Create residual upsampling layer (UpProjection)"""
        super(UpProject, self).__init__()
        self.up_conv = UnpoolAsConv(in_channels, out_channels, stride)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding='same')
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() #FIXME: inplace=True

    def forward(self, input_data, BN=True):
        # Branch 1
        # --------
        # Interleaving convolutions and output of 1st branch
        branch1_output = self.up_conv(self.size, input_data, ReLU=True, BN=True)

        # Convolution following the UpProjection on the 1st branch
        branch1_output = self.conv(branch1_output)
        if BN:
            branch1_output = self.norm(branch1_output)

        # Branch 2
        # --------
        # Interleaving convolutions and output of 2nd branch
        branch2_output = self.up_conv(self.size, input_data, ReLU=False)

        # Sum branches
        output = branch1_output + branch2_output
        
        # ReLU
        output = self.relu(output)

        return output

class UnpoolAsConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Model upconvolutions (unpooling + convolution) as interleaving feature
        maps of four convolutions (A,B,C,D). Building block for up-projections.
        """
        super(UnpoolAsConv, self).__init__()
        self.convA = nn.Conv2d(in_channels, out_channels, (3, 3), stride, padding=1)
        self.convB = nn.Conv2d(in_channels, out_channels, (2, 3), stride)
        self.convC = nn.Conv2d(in_channels, out_channels, (3, 2), stride)
        self.convD = nn.Conv2d(in_channels, out_channels, (2, 2), stride)
        self.norm  = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU() #FIXME: inplace=True

    def get_incoming_shape(self, incoming):
        """ Returns the incoming data shape """
        if isinstance(incoming, tf.Tensor):
            return incoming.get_shape().as_list()
        elif type(incoming) in [np.array, list, tuple]:
            return np.shape(incoming)
        else:
            raise Exception("Invalid incoming layer.")

    def interleave(self, tensors, axis):
        old_shape = self.get_incoming_shape(tensors[0])[1:]
        new_shape = [-1] + old_shape
        new_shape[axis] *= len(tensors)
        # return tf.reshape(tf.stack(tensors, axis + 1), new_shape)
        return torch.stack(tensors, axis + 1).view(*new_shape)

    def forward(self, input_data, ReLU=False, BN=True):
        # Convolution A (3x3)
        outA = self.convA(input_data)

        # Convolution B (2x3)
        padB = F.pad(input_data, (0, 0, 1, 0, 1, 1, 0, 0), mode='constant', value=0)
        outB = self.convB(padB)

        # Convolution C (3x2)
        padC = F.pad(input_data, (0, 0, 1, 1, 1, 0, 0, 0), mode='constant', value=0)
        outC = self.convC(padC)

        # Convolution D (2x2)
        padD = F.pad(input_data, (0, 0, 1, 0, 1, 0, 0, 0), mode='constant', value=0)
        outD = self.convD(padD)

        # Interleaving elements of the four feature maps
        L = self.interleave([outA, outB], axis=1)  # columns
        R = self.interleave([outC, outD], axis=1)  # columns
        Y = self.interleave([L, R], axis=2) # rows

        if BN:
            Y = self.norm(Y)

        if ReLU:
            Y = self.relu(Y)

        return Y
