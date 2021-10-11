#%%Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torchvision import models
import tensorflow.compat.v1 as tf
#sfrom numba import jit

from helpers import *

torch.backends.cudnn.deterministic=True
torch.manual_seed(1)
torch.cuda.empty_cache()

#%%Using a custom quality metric
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
frozen_graph = "models/IRQM4.pb"

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
with tf.gfile.GFile(frozen_graph, "rb") as fil:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(fil.read())

g2 = tf.Graph()
with g2.as_default():
    tf.import_graph_def(restored_graph_def, name='')
sess2 = tf.Session(graph=g2)

with tf.device('/gpu:0'):
    weights = sess2.run(['conv1/w:0', 'conv2/w:0', 'conv3/w:0', 'conv4/w:0', 'conv5/w:0', 'conv6/w:0', 'fc1/kernel:0', 'fc3/kernel:0'])
    biases = sess2.run(['conv1/b:0', 'conv2/b:0', 'conv3/b:0', 'conv4/b:0', 'conv5/b:0', 'conv6/b:0', 'fc1/bias:0', 'fc3/bias:0'])

#Quality tool
def IRQM(im):
    score_out = g2.get_tensor_by_name("fc3_1/BiasAdd:0")
    input_im = g2.get_tensor_by_name("TestImage:0")
    op4 = sess2.run(score_out, feed_dict={ input_im: im })
    return op4

#%%Define IRQM in pytorch
class IRQM_net(nn.Module):

    def get_gaussian_kernel(self, kernel_size=5, sigma=2, channels=1):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        
        return gaussian_filter

    def __init__(self, requires_grad=False):
        super(IRQM_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv2 = nn.Conv2d(16, 16, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv3 = nn.Conv2d(16, 32, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv4 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv5 = nn.Conv2d(32, 64, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv6 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        #self.conv7 = nn.Conv2d(128, 256, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        #self.conv8 = nn.Conv2d(256, 512, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.lin1 = nn.Linear(64, 64)
        self.lin2 = nn.Linear(64, 1)

        #Assign values to all weights
        self.conv1.weight = nn.Parameter(torch.Tensor(weights[0].transpose(3, 2, 0, 1)))
        self.conv2.weight = nn.Parameter(torch.Tensor(weights[1].transpose(3, 2, 0, 1)))
        self.conv3.weight = nn.Parameter(torch.Tensor(weights[2].transpose(3, 2, 0, 1)))
        self.conv4.weight = nn.Parameter(torch.Tensor(weights[3].transpose(3, 2, 0, 1)))
        self.conv5.weight = nn.Parameter(torch.Tensor(weights[4].transpose(3, 2, 0, 1)))
        self.conv6.weight = nn.Parameter(torch.Tensor(weights[5].transpose(3, 2, 0, 1)))
        #self.conv7.weight = nn.Parameter(torch.Tensor(weights[6].transpose(3, 2, 0, 1)))
        #self.conv8.weight = nn.Parameter(torch.Tensor(weights[7].transpose(3, 2, 0, 1)))
        self.lin1.weight = nn.Parameter(torch.Tensor(weights[6].transpose()))
        self.lin2.weight = nn.Parameter(torch.Tensor(weights[7].transpose()))

        #Assign values to all biases
        self.conv1.bias = nn.Parameter(torch.Tensor(biases[0]))
        self.conv2.bias = nn.Parameter(torch.Tensor(biases[1]))
        self.conv3.bias = nn.Parameter(torch.Tensor(biases[2]))
        self.conv4.bias = nn.Parameter(torch.Tensor(biases[3]))
        self.conv5.bias = nn.Parameter(torch.Tensor(biases[4]))
        self.conv6.bias = nn.Parameter(torch.Tensor(biases[5]))
        #self.conv7.bias = nn.Parameter(torch.Tensor(biases[6]))
        #self.conv8.bias = nn.Parameter(torch.Tensor(biases[7]))
        self.lin1.bias = nn.Parameter(torch.Tensor(biases[6]))
        self.lin2.bias = nn.Parameter(torch.Tensor(biases[7]))
        self.filt = self.get_gaussian_kernel()

    def forward(self, x):

        x -= self.filt(F.pad(x, (2, 2, 2, 2)))
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
        #x = F.elu(self.conv7(x))
        #x = F.elu(self.conv8(x))
        conv_feat = x.clone()
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = F.elu(self.lin1(x))
        x = self.lin2(x)
        return x #is an approximate offset we observe when switching from Tf to PyTorch

#Save quality network
feat_ext = IRQM_net()
torch.save(feat_ext.state_dict(), 'models/IRQM4.pth')
print('Network saved!!')

#%%Test networks
dir = 'C:/Users/z003zv1a/Desktop/Code/reconstucted_ims/Images/Noise comparison/clinical/train/WFBP/'
phantom = dicom_read(dir)

#Include IRQM as critic, other already present
critic = IRQM_net()
critic.load_state_dict(torch.load('models/IRQM4.pth'))
critic.eval()

#Compare and print inference scores
score_comparison = np.zeros((20, 2))
for i in range(0, 20):
    score_comparison[i, 0] = IRQM(np.expand_dims(np.expand_dims(phantom[i,:,:], axis=2), axis=0))[0,0]
    score_comparison[i, 1] = critic(torch.Tensor(phantom[i,:,:]).view(1, 1, 256, 256)).detach().numpy()[0,0]

print(score_comparison)

#%%
