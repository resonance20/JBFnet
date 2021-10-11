#%%Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.model_selection import train_test_split
import time
from pathos.pools import ParallelPool
from skimage.measure import compare_ssim

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torchvision import models

from helpers import *

torch.backends.cudnn.deterministic=True
torch.manual_seed(1)
torch.cuda.empty_cache()

#%%Load massive train data
dir = 'C:/Users/z003zv1a/Documents/Images/'
ltime = time.time()
phantom_list = []
noisy_phantom_list = []

for no, patient in enumerate(os.listdir(dir)):
    if no>=10:
        break

    newdir = dir + patient
    if os.path.isdir(newdir):

        for recon in os.listdir(newdir):
            newdir2 = newdir + '/' + recon
            if os.path.isdir(newdir2) and 'WFBP' in recon:

                for doselevel in os.listdir(newdir2):
                    finaldir = newdir2 + '/' + doselevel
                    if (os.path.isdir(finaldir)) and '100' in doselevel:
                        phantom_list.append(finaldir + '/')
                        #phantom_new = dicom_read(finaldir + '/')
                        #phantom = np.concatenate((phantom, phantom_new, phantom_new, phantom_new, phantom_new), axis = 0)

                    if (os.path.isdir(finaldir)) and '100' not in doselevel and 'High' not in doselevel:
                        noisy_phantom_list.append(finaldir + '/')
                        #phantom_new = dicom_read(finaldir + '/')
                        #noisy_phantom = np.concatenate((noisy_phantom, phantom_new), axis = 0) 
pool = ParallelPool(8)
noisy_phantom_raw = pool.map(dicom_read, noisy_phantom_list)
phantom_raw = pool.map(dicom_read, phantom_list) 
pool.close()
pool.join()

print('Data loaded!')
print('Time to load: %f seconds'%(round(time.time() - ltime, 2)) )

#Convert to non-overlapping 3D slabs
noisy_phantom = np.concatenate([vol for vol in noisy_phantom_raw], axis=0)
phantom = np.concatenate([np.concatenate([vol, vol, vol, vol], axis=0) for vol in phantom_raw], axis=0)
"""
print('Converting to slabs...')
ltime = time.time()
thickness = 8

if phantom.shape[0]%thickness is not 0:
    padding_depth = thickness - phantom.shape[0]%thickness
    phantom = np.concatenate((phantom, np.zeros([padding_depth, phantom.shape[1], phantom.shape[2]])), axis = 0)
    noisy_phantom = np.concatenate((noisy_phantom, np.zeros([padding_depth, noisy_phantom.shape[1], noisy_phantom.shape[2]])), axis = 0)
phantom = np.reshape(phantom, (-1, thickness, phantom.shape[1], phantom.shape[2]))
noisy_phantom = np.reshape(noisy_phantom, (-1, thickness, noisy_phantom.shape[1], noisy_phantom.shape[2]))
"""
print(np.shape(phantom))
print(np.shape(noisy_phantom))

print('Time to convert: %f seconds'%(round(time.time() - ltime, 2)) )

#%%Find 3d Sobel gradient image
def gradIm(image):
    
    filtx = torch.Tensor([[[-1, 0, 1], [0, 0, 2], [-2, 0, 1]], 
    [[-2, 0, -2], [-4, 0, 4], [-2, 0, 2]], 
    [[-1, 0, 1], [0, 0, 2], [-2, 0, 1]]]).cuda()

    filty = torch.Tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
    [[2, 4, 2], [0, 0, 0], [-2, -4, -2]], 
    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]).cuda()

    filtz = torch.Tensor([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], 
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 
    [[1, 2, 1], [2, 4, 2], [1, 2, 1]]]).cuda()

    Gx = F.conv3d(image, filtx.view((1,1,3,3,3)))
    Gy = F.conv3d(image, filty.view((1,1,3,3,3)))
    Gz = F.conv3d(image, filtz.view((1,1,3,3,3)))

    del filtx
    del filty
    del filtz
    torch.cuda.empty_cache()

    return Gx, Gy, Gz

#Gradient loss
def gdl(im1, im2):

    Gx1, Gy1, Gz1 = gradIm(im1)
    Gx2, Gy2, Gz2 = gradIm(im2)

    loss_x = torch.abs(Gx1) - torch.abs(Gx2)
    loss_y = torch.abs(Gy1) - torch.abs(Gy2)
    loss_z = torch.abs(Gz1) - torch.abs(Gz2)

    return loss_x**2 + loss_y**2 + loss_z**2

#%%Find 2d Sobel gradient image
def gradIm2d(image):
    
    filtx = torch.Tensor([[1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]]).cuda()

    filty = torch.Tensor([[1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]]).cuda()

    Gx = F.conv2d(image, filtx.view((1,1,3,3)))
    Gy = F.conv2d(image, filty.view((1,1,3,3)))

    del filtx
    del filty
    torch.cuda.empty_cache()

    return Gx, Gy

#Gradient loss
def gdl2d(im1, im2):

    Gx1, Gy1 = gradIm2d(im1)
    Gx2, Gy2 = gradIm2d(im2)

    loss_x = torch.abs(Gx1) - torch.abs(Gx2)
    loss_y = torch.abs(Gy1) - torch.abs(Gy2)

    return torch.mean(loss_x**2 + loss_y**2)

#%%Convert dataset to torch
noisy_train, noisy_test, phantom_train, phantom_test = train_test_split(noisy_phantom, phantom)

tensor_phantom = torch.stack([torch.Tensor(i).unsqueeze(0) for i in phantom_train])
tensor_noisy = torch.stack([torch.Tensor(i).unsqueeze(0) for i in noisy_train])
tensor_phantom_test = torch.stack([torch.Tensor(i).unsqueeze(0) for i in phantom_test])
tensor_noisy_test = torch.stack([torch.Tensor(i).unsqueeze(0) for i in noisy_test])

dataset = torch.utils.data.TensorDataset(tensor_phantom, tensor_noisy)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
dataset = torch.utils.data.TensorDataset(tensor_phantom_test, tensor_noisy_test)
valloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
print('Dataset torchified!')

#%%3D denoiser
class enc(nn.Module):
    def __init__(self):
        super(enc, self).__init__()
        self.conv1 = nn.Conv3d(1, 96, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))
        self.conv2 = nn.Conv3d(96, 96, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))
        self.conv3 = nn.Conv3d(96, 96, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))
        self.conv4 = nn.Conv3d(96, 96, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))
        self.upconv4 = nn.ConvTranspose3d(96, 96, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))
        self.upconv3 = nn.ConvTranspose3d(96, 96, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))
        self.upconv2 = nn.ConvTranspose3d(96, 96, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))
        self.upconv1 = nn.ConvTranspose3d(96, 1, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))

    def forward(self, x):
        im = x.clone()
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        mid = x.clone()
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.upconv4(x))
        x = F.leaky_relu(self.upconv3(x) + mid)
        x = F.leaky_relu(self.upconv2(x))
        return (self.upconv1(x) + im)

#2D Denoiser
class enc2d(nn.Module):
    def __init__(self):
        super(enc2d, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv2 = nn.Conv2d(96, 96, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv3 = nn.Conv2d(96, 96, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv4 = nn.Conv2d(96, 96, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.upconv4 = nn.ConvTranspose2d(96, 96, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.upconv3 = nn.ConvTranspose2d(96, 96, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.upconv2 = nn.ConvTranspose2d(96, 96, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.upconv1 = nn.ConvTranspose2d(96, 1, kernel_size = (3, 3), stride = 1, padding = (1, 1))

    def forward(self, x):
        im = x.clone()
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        mid = x.clone()
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.upconv4(x))
        x = F.leaky_relu(self.upconv3(x) + mid)
        x = F.leaky_relu(self.upconv2(x))
        return (self.upconv1(x) + im)

#%%Loss functions and optimiser
try:
    del E2
    del loss_network
    torch.cuda.empty_cache()
except NameError:
    print('Tensors not initialised')
E2 = enc2d().cuda()
loss_network = IRQM_net().cuda()
loss_network.load_state_dict(torch.load('denoise_dl/models/IRQM.pth'))
loss_network.eval()
optim_AE1_enc = optim.Adam(E2.parameters(), lr = 1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim_AE1_enc, gamma = 0.95)

model_parameters = filter(lambda p: p.requires_grad, E2.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Number of trainable parameters: ' +str(params))

#%%Train branch
for epoch in range(5):
    running_loss = 0
    val_loss = 0
    E2.train() 

    start = time.time()

    #Train step
    for i, data in enumerate(dataloader, 0):

        #Load data and initialise optimisers
        phantom_in, noisy_in = data        
        phantom_in = phantom_in.cuda()
        noisy_in = noisy_in.cuda()

        optim_AE1_enc.zero_grad()

        #Forward pass
        outputs1 = E2(noisy_in)
        with torch.no_grad():
            _, gt_feat = loss_network(phantom_in)
            _, gt_pred = loss_network(outputs1)

        #Calculate perceptual loss
        loss = nn.MSELoss()(outputs1, phantom_in)
        #gd_loss = gdl2d(outputs1, phantom_in)
        irqm_loss = torch.mean((gt_feat - gt_pred)**2)
        
        loss =  (loss) + irqm_loss

        #Backprop and update for AE1
        loss.backward()
        optim_AE1_enc.step()

        # print statistics
        running_loss += loss.item()

        #Clear memory
        #del phantom_in
        #del noisy_in
        #torch.cuda.empty_cache()

    #Val step
    E2.eval()
    torch.cuda.empty_cache()

    for j, data in enumerate(valloader, 0):

        #Load validation data
        phantom_in, noisy_in = data
        phantom_in = phantom_in.cuda()
        noisy_in = noisy_in.cuda()

        #Forward pass in the validation phase
        with torch.no_grad():
            outputs1 = E2(noisy_in)
            _, gt_feat = loss_network(phantom_in)
            _, gt_pred = loss_network(outputs1)

        #Calculate perceptual loss
        loss = nn.MSELoss()(outputs1, phantom_in) 
        #gd_loss = gdl2d(outputs1, phantom_in)
        irqm_loss = torch.mean((gt_feat - gt_pred)**2)

        loss =  (loss) + irqm_loss

        # print statistics
        val_loss += loss.item()

        #Clear memory
        #del phantom_in
        #del noisy_in
        #torch.cuda.empty_cache()

    print('[%d, %5d] train_loss: %.3f val_loss: %.3f' %
            (epoch + 1, i + 1, running_loss / int(i+1), val_loss / int(j+1) ))
    running_loss = 0.0
    val_loss = 0.0
    end = time.time()
    print('Time taken for epoch: '+str(end-start)+' seconds')
    scheduler.step()

    if (epoch+1)%5 == 0:
        print('Saving model...')
        torch.save(E2.state_dict(), 'denoise_dl/models/2dIRQM/GTdenoiseAE_'+str(epoch+1)+'.pth')

print('Training complete!')

#%%Load test data
dir = 'C:/Users/z003zv1a/Desktop/Code/reconstucted_ims/Images/Noise comparison/clinical/test/WFBP/'
phantom_test_set = dicom_read(dir)
#phantom_test_set = convert_3d_form(phantom_test_set)

dir = 'C:/Users/z003zv1a/Desktop/Code/reconstucted_ims/Images/Noise comparison/clinical/test/WFBP_25/'
noisy_phantom_test = dicom_read(dir)
#noisy_phantom_test = convert_3d_form(noisy_phantom_test)

dir = 'C:/Users/z003zv1a/Desktop/Code/reconstucted_ims/Images/Noise comparison/clinical/test/WFBP/'
noisy_phantom_denoise = dicom_read(dir)
#noisy_phantom_denoise = convert_3d_form(noisy_phantom_denoise)

#%%Inference test
indno = 224
test_im = noisy_phantom_test[indno]
test_im_gt = phantom_test_set[indno]
test_slab = noisy_phantom_test[indno-4 : indno+4]
custom_enc = enc().cuda()
custom_enc.load_state_dict(torch.load('models/3dCNN/GTdenoiseAE_5.pth'))
custom_enc.eval()

custom_enc2 = enc().cuda()
custom_enc2.load_state_dict(torch.load('models/3dIRQM/GTdenoiseAE_5.pth'))
custom_enc2.eval()

custom_enc2d = enc2d().cuda()
custom_enc2d.load_state_dict(torch.load('models/2dCNN/GTdenoiseAE_5.pth'))
custom_enc.eval()

custom_enc2d2 = enc2d().cuda()
custom_enc2d2.load_state_dict(torch.load('models/2dIRQM/GTdenoiseAE_5.pth'))
custom_enc2d2.eval()

input = torch.Tensor(test_slab).unsqueeze(0).unsqueeze(0)
output = custom_enc(input.cuda())
denoised_slab = output.cpu().detach().numpy()[0,0,:,:]

input = torch.Tensor(test_slab).unsqueeze(0).unsqueeze(0)
output = custom_enc2(input.cuda())
denoised_slab_irqm = output.cpu().detach().numpy()[0,0,:,:]

input = torch.Tensor(test_im).unsqueeze(0).unsqueeze(0)
output = custom_enc2d(input.cuda())
denoised_im = output.cpu().detach().numpy()[0,0,:,:]

input = torch.Tensor(test_im).unsqueeze(0).unsqueeze(0)
output = custom_enc2d2(input.cuda())
denoised_im_irqm = output.cpu().detach().numpy()[0,0,:,:]

plt.rcParams.update({'font.size': 30})
fig, axes = plt.subplots(2, 3, figsize =(30, 20))
axes[0, 0].axis('off')
axes[0, 0].imshow(window(test_im), cmap = 'gray')
axes[0, 0].set_title('(a) 25% Dose')
axes[0, 1].axis('off')
axes[0, 1].imshow(window(denoised_im), cmap = 'gray')
axes[0, 1].set_title('(b) 2D Denoiser (MSE)')
axes[0, 2].axis('off')
axes[0, 2].imshow(window(denoised_im_irqm), cmap = 'gray')
axes[0, 2].set_title('(c) 2D Denoiser (Perc.)')
axes[1, 0].axis('off')
axes[1, 0].imshow(window(test_im_gt), cmap = 'gray')
axes[1, 0].set_title('(d) Full Dose')
axes[1, 1].axis('off')
axes[1, 1].imshow(window(denoised_slab[4, :, :]), cmap = 'gray')
axes[1, 1].set_title('(e) 3D Denoiser (MSE)')
axes[1, 2].axis('off')
axes[1, 2].imshow(window(denoised_slab_irqm[4, :, :]), cmap = 'gray')
axes[1, 2].set_title('(f) 3D Denoiser (Perc.)')

#%%Calculate quantitative results
psnr_mat = []
ssim_mat = []

custom_enc = enc().cuda()
custom_enc.load_state_dict(torch.load('models/3dCNN/GTdenoiseAE_5.pth'))
custom_enc.eval()

custom_enc2 = enc().cuda()
custom_enc2.load_state_dict(torch.load('models/3dIRQM/GTdenoiseAE_5.pth'))
custom_enc2.eval()

custom_enc2d = enc2d().cuda()
custom_enc2d.load_state_dict(torch.load('models/2dCNN/GTdenoiseAE_5.pth'))
custom_enc.eval()

custom_enc2d2 = enc2d().cuda()
custom_enc2d2.load_state_dict(torch.load('models/2dIRQM/GTdenoiseAE_5.pth'))
custom_enc2d2.eval()

for i in range(5, noisy_phantom_test.shape[0] - 5):
    im = noisy_phantom_test[i-4 : i+4 ]
    input = torch.Tensor(im).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output1 = custom_enc(input.cuda())
        output2 = custom_enc2(input.cuda())
        output3 = custom_enc2d(input[:,:,4,:,:].cuda())
        output4 = custom_enc2d2(input[:,:,4,:,:].cuda())

    denoised_im1 = output1.cpu().detach().numpy()[0,0,4,:,:]
    denoised_im2 = output2.cpu().detach().numpy()[0,0,4,:,:]
    denoised_im3 = output3.cpu().detach().numpy()[0,0,:,:]
    denoised_im4 = output4.cpu().detach().numpy()[0,0,:,:]

    psnr1 = psnr(denoised_im1, phantom_test_set[i])
    psnr2 = psnr(denoised_im2, phantom_test_set[i])
    psnr3 = psnr(denoised_im3, phantom_test_set[i])
    psnr4 = psnr(denoised_im4, phantom_test_set[i])

    ssim1 = compare_ssim(denoised_im1, phantom_test_set[i], data_range=4096)
    ssim2 = compare_ssim(denoised_im2, phantom_test_set[i], data_range=4096)
    ssim3 = compare_ssim(denoised_im3, phantom_test_set[i], data_range=4096)
    ssim4 = compare_ssim(denoised_im4, phantom_test_set[i], data_range=4096)

    ssim_mat.append([ssim1, ssim2, ssim3, ssim4])
    psnr_mat.append([psnr1, psnr2, psnr3, psnr4])
    
psnr_mat = np.array(psnr_mat)
ssim_mat = np.array(ssim_mat)
np.savez("ECR.npz", psnr_mat = psnr_mat, ssim_mat = ssim_mat)

#print("Mean PSNR: " + str(np.mean(np.array(psnr_mat, np.float32))))
#print("Mean SSIM: " + str(np.mean(np.array(ssim_mat, np.float32))))

#%%Evaluate
