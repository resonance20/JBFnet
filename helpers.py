import numpy as np
import os
import pydicom
import cv2
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import math
from scipy.spatial import distance
from skimage.measure import compare_ssim
from pathos.pools import ParallelPool
from pydicom.datadict import DicomDictionary, keyword_dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torchvision import models, transforms as T

#Mean squared error
def mse(img1, img2):
    return np.mean((img1-img2)**2)

#Peak signal to noise ratio
def psnr(bw_noisy, bw_ref):
    mse_val = mse(bw_noisy, bw_ref)
    if mse_val == 0:
        return 100
    psnr = 20 * np.log10(4096/np.sqrt(mse_val))
    return psnr

#Slice wise structural similarity
def ssim(img1, img2):
    result = np.zeros(img1.shape[0])
    for i in range(img1.shape[0]):
        result[i] = compare_ssim(img1[i, :, :], img2[i, :, :], data_range=4096)
    return np.mean(result)

#Function to read a DICOM folder
def dicom_read(path):
  
    #Load lis t of dicom files
    list_files = os.listdir(path)
    list_dicom = []
    for file in list_files:
        if file.endswith('.dcm') or file.endswith('.IMA'):
            list_dicom.append(file)

    #Find reference values
    RefDs = pydicom.read_file(path + list_dicom[0])
    #const_pixel_dims = (len(list_dicom), RefDs.Rows, RefDs.Columns, )
    const_pixel_dims = (len(list_dicom), 256, 256)
    
    #Create array and load values
    dicom_array = np.zeros(const_pixel_dims)
    for file in list_dicom:
        ds = pydicom.dcmread(path + file)
        im = np.array(ds.pixel_array, np.int16)
        dicom_array[list_dicom.index(file),:,:] = cv2.resize(im, (256, 256))
    
    return dicom_array

#Function to read a DICOM folder
def dicom_read_list(path, oneRot = False):
  
    #Load lis t of dicom files
    list_files = os.listdir(path)
    list_dicom = []
    for file in list_files:
        if file.endswith('.dcm') or file.endswith('.IMA'):
            list_dicom.append(os.path.join(path, file))
    if oneRot:
        lead_proj = pydicom.dcmread(list_dicom[0])
        numberOfProjNeeded = int.from_bytes(lead_proj['0x70331013'].value, 'little')
        list_dicom = list_dicom[:(numberOfProjNeeded*3)]
    pool = ParallelPool(16)
    list_scans = pool.map(pydicom.dcmread, list_dicom)
    pool.close()
    pool.join()
    pool.clear()
    update_pydicom_dict('C:/Users/z003zv1a/Downloads/dict.txt')

    return list_scans

#Function to update dictionary
def update_pydicom_dict(dir):
    f = open(dir, 'r')
    list_tags = f.readlines()
    f.close()
    for tag in list_tags:
        tag_spl = tag.split('\t')#Split delimiters
        tag_spl[-1] = tag_spl[-1][:-1]#Remove newline charcter
        tag_spl[0] = '0x' + tag_spl[0][1:5] + tag_spl[0][6:10]
        DicomDictionary.update({tag_spl[0]: (tag_spl[1], tag_spl[2], tag_spl[3])})
        keyword_dict.update({tag_spl[2]: tag_spl[0]})
    print('Dictionary has been updated!')

#%%Custom class for inference network
class IRQM_net(nn.Module):

    def __init__(self, requires_grad=False):
        super(IRQM_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv2 = nn.Conv2d(16, 16, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv3 = nn.Conv2d(16, 32, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv4 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv5 = nn.Conv2d(32, 64, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv6 = nn.Conv2d(64, 128, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv7 = nn.Conv2d(128, 256, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv8 = nn.Conv2d(256, 512, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.lin1 = nn.Linear(512, 512)
        self.lin2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
        x = F.elu(self.conv7(x))
        x = F.elu(self.conv8(x))
        conv_feat = x.clone()
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = F.elu(self.lin1(x))
        x = self.lin2(x)
        return torch.sigmoid(x[:, 0]), conv_feat

#Model iterative reconstruction as a  feed forward CNN
class redcnn(nn.Module):
    def __init__(self):
        super(redcnn, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size = (3, 3, 3), stride = 1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size = (3, 3, 3), stride = 1)
        self.conv3 = nn.Conv3d(32, 32, kernel_size = (3, 3, 3), stride = 1)
        self.conv4 = nn.Conv3d(32, 32, kernel_size = (3, 3, 3), stride = 1)
        self.pc1 = nn.Conv3d(64, 32, kernel_size = (1, 1, 1), stride = 1)
        self.pc2 = nn.Conv3d(64, 32, kernel_size = (1, 1, 1), stride = 1)
        self.pc3 = nn.Conv3d(64, 32, kernel_size = (1, 1, 1), stride = 1)
        self.upconv4 = nn.ConvTranspose3d(32, 32, kernel_size = (1, 3, 3), stride = 1)
        self.upconv3 = nn.ConvTranspose3d(32, 32, kernel_size = (1, 3, 3), stride = 1)
        self.upconv2 = nn.ConvTranspose3d(32, 32, kernel_size = (1, 3, 3), stride = 1)
        self.upconv1 = nn.ConvTranspose3d(32, 1, kernel_size = (1, 3, 3), stride = 1)

    def forward(self, x):

        width = x.shape[2] - 8
        #Conv
        x = F.leaky_relu(self.conv1(x))
        im1 = x.clone()[:,:,4:4 + width,:,:]
        x = F.leaky_relu(self.conv2(x))
        im2 = x.clone()[:,:,3:3 + width,:,:]
        x = F.leaky_relu(self.conv3(x))
        im3 = x.clone()[:,:,2:2 + width,:,:]
        x = F.leaky_relu(self.conv4(x))
        #Deconv
        x = F.leaky_relu(self.upconv4(x))
        x = F.leaky_relu(self.pc1(torch.cat((x, im3), 1) ))
        x = F.leaky_relu(self.upconv3(x))
        x = F.leaky_relu(self.pc2(torch.cat((x, im2), 1) ))
        x = F.leaky_relu(self.upconv2(x))
        x = F.leaky_relu(self.pc3(torch.cat((x, im1), 1) ))
        return self.upconv1(x)

#Generator
class gen(nn.Module):
    def __init__(self):
        super(gen, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size = (3, 3, 3), stride = 1, padding = (0, 1, 1))
        self.conv2 = nn.Conv3d(32, 32, kernel_size = (3, 3, 3), stride = 1, padding = (0, 1, 1))
        self.conv3 = nn.Conv3d(32, 64, kernel_size = (3, 3, 3), stride = 1, padding = (0, 1, 1))
        self.conv4 = nn.Conv3d(64, 64, kernel_size = (3, 3, 3), stride = 1, padding = (0, 1, 1))
        self.conv5 = nn.Conv3d(64, 128, kernel_size = (3, 3, 3), stride = 1, padding = (0, 1, 1))
        self.conv6 = nn.Conv3d(128, 128, kernel_size = (3, 3, 3), stride = 1, padding = (0, 1, 1))
        self.conv7 = nn.Conv3d(128, 1, kernel_size = (3, 3, 3), stride = 1, padding = (0, 1, 1))
       
    def forward(self, x):
        im = x.clone()[:,:,7:8,:,:]
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = self.conv7(x)
        return im - x

#sigmoid
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

#Function to window an image
def window(im, centre = 40, width = 300, slope = 1, bias = -1024):

    im_modded = im*slope + bias
    im_modded[im_modded>(centre + width/2)] = centre + width/2
    im_modded[im_modded<(centre - width/2)] = centre - width/2

    return im_modded