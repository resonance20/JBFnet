#%%Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.model_selection import train_test_split
from skimage.restoration import denoise_nl_means
from scipy.spatial import distance
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data
import torch.utils.checkpoint as checkpoint
from torch.autograd import Variable
from pathos.multiprocessing import ProcessPool
from pathos.pp import ParallelPool
from EFLoss import EFLoss
import nlmeans_cpp

from helpers import *

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.backends.cudnn.deterministic=True
torch.manual_seed(1)
torch.cuda.empty_cache()

#%%Load massive train data
ltime = time.time()
dir = 'C:/Users/z003zv1a/Documents/Images/Old/'
#phantom = np.zeros((1, 256, 256))
#noisy_phantom = np.zeros((1, 256, 256))

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
pool = ParallelPool(nodes = 8)
noisy_phantom_raw = pool.map(dicom_read, noisy_phantom_list)
phantom_raw = pool.map(dicom_read, phantom_list) 
pool.close()
pool.join()
pool.clear()

noisy_phantom = np.concatenate([vol for vol in noisy_phantom_raw], axis=0)
phantom = np.concatenate([np.concatenate([vol, vol, vol, vol], axis=0) for vol in phantom_raw], axis=0)

print('Data loaded!')
print('Time to load: %f seconds'%(round(time.time() - ltime, 2)) )
print(np.shape(phantom))
print(np.shape(noisy_phantom))

#Convert to non-overlapping 3D slabs
print('Converting to slabs...')
thickness = 19
if phantom.shape[0]%thickness is not 0:
    padding_depth = thickness - phantom.shape[0]%thickness
    phantom = np.concatenate((phantom, np.zeros([padding_depth, phantom.shape[1], phantom.shape[2]])), axis = 0)
    noisy_phantom = np.concatenate((noisy_phantom, np.zeros([padding_depth, noisy_phantom.shape[1], noisy_phantom.shape[2]])), axis = 0)
phantom = np.reshape(phantom, (-1, thickness, phantom.shape[1], phantom.shape[2]))
noisy_phantom = np.reshape(noisy_phantom, (-1, thickness, noisy_phantom.shape[1], noisy_phantom.shape[2]))

print(np.shape(phantom))
print(np.shape(noisy_phantom))

#Convert to 3D patches
print('Converting to patches...')
phantom = phantom.reshape(phantom.shape[0], thickness, 4, 64, 4, 64).swapaxes(3, 4). \
reshape(phantom.shape[0], thickness, -1, 64, 64).swapaxes(1, 2).reshape(-1, thickness, 64, 64)
noisy_phantom = noisy_phantom.reshape(noisy_phantom.shape[0], thickness, 4, 64, 4, 64).swapaxes(3, 4). \
reshape(noisy_phantom.shape[0], thickness, -1, 64, 64).swapaxes(1, 2).reshape(-1, thickness, 64, 64)

print(np.shape(phantom))
print(np.shape(noisy_phantom))

#%%Dummy data
#phantom = np.random.randint(0, 4096,size=(80, 7, 64, 64))
#noisy_phantom = np.random.randint(0, 4096,size=(80, 15, 64, 64))

#%%Convert dataset to torch
noisy_train, noisy_test, phantom_train, phantom_test = train_test_split(noisy_phantom, phantom)

tensor_phantom = torch.from_numpy(phantom_train).unsqueeze(1).float()
tensor_noisy = torch.from_numpy(noisy_train).unsqueeze(1).float()
tensor_phantom_test = torch.from_numpy(phantom_test).unsqueeze(1).float()
tensor_noisy_test = torch.from_numpy(noisy_test).unsqueeze(1).float()

dataset = torch.utils.data.TensorDataset(tensor_phantom, tensor_noisy)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True)
dataset = torch.utils.data.TensorDataset(tensor_phantom_test, tensor_noisy_test)
valloader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=True)
print('Dataset torchified!')

#%%Architecture segments

#NL Means
class NLMeansCpp(autograd.Function):

    @staticmethod
    def forward(ctx, x, p = 3, s = 7):
        z_stat = nlmeans_cpp.forward(x, p, s)
        return z_stat

    @staticmethod
    def backward(ctx, grad):
        grad_input = nlmeans_cpp.backward(grad)
        return grad_input

#Joint Bilateral Filtering block
class JBF_block(nn.Module):

    def layers(self):
        lay = []
        num = int((self.n - 3)/2)
        for _ in range(num):
            lay.append( nn.Conv3d(1, 1, kernel_size = (3, 3, 3), stride = 1) )
            lay.append(nn.ReLU())
        return nn.Sequential(*lay)
    
    def __init__(self, n):
        super(JBF_block, self).__init__()
        self.n = n
        self.range_coeffecients = self.layers()
        self.domain_coeffecients = self.layers()
        
    def forward(self, x, domain_neighbor, guide_im):
        
        #Estimate range filter coeffecients
        guide_im = guide_im.unfold(2, self.n, 1).unfold(3, self.n, 1).unfold(4, self.n, 1)\
            .reshape(-1, 1, self.n, self.n, self.n)
        guide_im -= guide_im[:, 0, int(self.n/2), int(self.n/2), int(self.n/2)].view(guide_im.shape[0], 1, 1, 1, 1)
        guide_im  = torch.abs(guide_im)
        range_kernel = self.range_coeffecients(guide_im)

        #Estimate domain filter coeffecients
        centre_x = math.floor(domain_neighbor.shape[2]/2) - int(self.n/2)
        centre_y = math.floor(domain_neighbor.shape[2]/2) - int(self.n/2) + self.n
        domain_kernel = self.domain_coeffecients(domain_neighbor[:, :, centre_x:centre_y, centre_x:centre_y, centre_x:centre_y])

        #Apply bilateral filter
        weights = (domain_kernel*range_kernel) + 1e-10
        
        n2 = self.n - 2
        x_trun = x[:, :, int(n2/2) : x.shape[2] - int(n2/2), int(n2/2) : x.shape[3] - int(n2/2), int(n2/2) : x.shape[4] - int(n2/2)]
        x_unf = x_trun.unfold(2, 3, 1).unfold(3, 3, 1).unfold(4, 3, 1).reshape(-1, 1, 3, 3, 3)
        weighted_x = x_unf*weights

        filtered_im = weighted_x.view(weighted_x.shape[0], 1, -1).sum(2) / weights.view(weights.shape[0], 1, -1).sum(2)
        return filtered_im.view(x.shape[0], 1, x.shape[2] - self.n + 1, x.shape[3] - self.n + 1, x.shape[4] - self.n + 1)

#Multiscale Joint Bilateral Filtering block
class mixed_JBF_block(nn.Module):
    
    def __init__(self):
        super(mixed_JBF_block, self).__init__()
        #self.JBF_block1 = JBF_block(3)
        #self.JBF_block2 = JBF_block(5)
        self.JBF_block3 = JBF_block(7)
        self.JBF_block4 = JBF_block(9)
        self.collate = nn.Conv3d(3, 1, kernel_size=(1, 1, 1), stride = 1)
        
    def forward(self, x, domain_neighbor, guide_im):
        
        #Filter image at different scales
        #filter_3 = F.relu(self.JBF_block1(x, domain_neighbor, guide_im))
        #filter_5 = F.leaky_relu(self.JBF_block2(x, domain_neighbor, guide_im))
        filter_7 = F.leaky_relu(self.JBF_block3(x, domain_neighbor, guide_im))
        filter_9 = F.leaky_relu(self.JBF_block4(x, domain_neighbor, guide_im))
        """
        fig, axes = plt.subplots(1, 3, figsize=(30,10))
        for ax in axes:
            ax.axis('off')
        axes[0].imshow(x.cpu().detach().numpy()[0, 0, math.ceil(x.shape[2]/2)], cmap='gray')
        #axes[1].imshow(filter_5.cpu().detach().numpy()[0, 0, math.ceil(filter_5.shape[2]/2)], cmap='gray')
        axes[1].imshow(filter_7.cpu().detach().numpy()[0, 0, math.ceil(filter_7.shape[2]/2)], cmap='gray')
        axes[2].imshow(filter_9.cpu().detach().numpy()[0, 0, math.ceil(filter_9.shape[2]/2)], cmap='gray')
        plt.show()
        input("w8")
        """
        x_crop = x[:, :, 4:-4, 4:-4, 4:-4]

        #nm_3 = F.relu(x_crop - filter_3[:, :, 3:-3, 3:-3, 3:-3])
        #nm_5 = F.leaky_relu(x_crop - filter_5[:, :, 2:-2, 2:-2, 2:-2])
        #nm_7 = F.leaky_relu(x_crop - filter_7[:, :, 1:-1, 1:-1, 1:-1])
        #nm_9 = F.leaky_relu(x_crop - filter_9)

        x_nm = F.leaky_relu(torch.cat((x_crop, filter_7[:, :, 1:-1, 1:-1, 1:-1], filter_9), dim = 1))
        x_nm = F.leaky_relu(self.collate(x_nm))

        return x_nm#F.relu(x_crop - x_nm)

#%%JBF net architecture
class JBF_net(nn.Module):
    
    def __init__(self):
        super(JBF_net, self).__init__()
        #Denoising
        self.JBF_block1 = mixed_JBF_block()
        self.JBF_block2 = mixed_JBF_block()
        self.skip1 = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride = 1, padding = (0, 1, 1), bias = False)
        self.skip2 = nn.Conv3d(1, 1, kernel_size=(1, 3, 3), stride = 1, padding = (0, 1, 1), bias = False)
        
    def forward(self, x, spat_kernel):

        depth = 4
        guide_im = NLMeansCpp.apply(x)

        x_filt = F.relu(self.JBF_block1(x, spat_kernel, guide_im))
        x_rem = F.leaky_relu(self.skip1(x_filt - x[:, :, depth:x.shape[2] - depth, depth:x.shape[3] - depth, depth:x.shape[4] - depth]) \
                       * (x_filt - x[:, :, depth:x.shape[2] - depth, depth:x.shape[3] - depth, depth:x.shape[4] - depth]))
        x = F.relu(x_filt + x_rem)
        f1 = x.clone()
        
        guide_im = guide_im[:, :, depth:guide_im.shape[2] - depth, depth:guide_im.shape[3] - depth, depth:guide_im.shape[4] - depth]
        x_filt = F.relu(self.JBF_block2(x, spat_kernel, guide_im))
        x_rem = F.leaky_relu(self.skip2(x_filt - x[:, :, depth:x.shape[2] - depth, depth:x.shape[3] - depth, depth:x.shape[4] - depth]) \
                       * (x_filt - x[:, :, depth:x.shape[2] - depth, depth:x.shape[3] - depth, depth:x.shape[4] - depth]))
        x = F.relu(x_filt + x_rem)
        
        return x, f1

#%%Loss functions and optimiser
try:
    del JBFnet
    torch.cuda.empty_cache()
except NameError:
    print('Tensors not initialised')
JBFnet = JBF_net().cuda()
#JBFnet.load_state_dict(torch.load('models/JBFnet_multiscale_1.pth'))

den_params = list(filter(lambda p: 'denoiser' in p[0], JBFnet.named_parameters()))
den_params = [p[1] for p in den_params]
#net_params = list(filter(lambda p: 'denoiser' not in p[0], JBFnet.named_parameters()))
#net_params = [p[1] for p in net_params]
optimiser_main = optim.Adam(JBFnet.parameters(), lr = 1e-4)
#optimiser_prior = optim.Adam(den_params, lr = 1e-4)

model_parameters = filter(lambda p: p.requires_grad, JBFnet.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print('Number of trainable parameters: ' +str(params))

#%%Pre train prior
def comb_loss(im1, im2):
    return  nn.L1Loss()(im1, im2) + EFLoss()(im1, im2) #+ nn.MSELoss()(im1, im2)

for epoch in range(10):
    running_loss = 0
    val_loss = 0
 
    #Precompute spatial kernel
    dim = 9
    spat_kernel = torch.zeros(1, 1, dim, dim, dim).cuda()
    for a in range(0, dim):
        for b in range(0, dim):
            for c in range(0, dim):
                spat_kernel[0, 0, a, b, c] = torch.Tensor( [distance.euclidean((a, b, c), (4, 4, 4)) ] )

    JBFnet.train() 
    start = time.time()

     #Train step
    for i, data in enumerate(dataloader, 0):
        stime = time.time()
        phantom_in, noisy_in = data
        
        #Rescale
        noisy_in = noisy_in/4096
        phantom_in = phantom_in/4096
        
        noisy_in = noisy_in.cuda()
        noisy_in.requires_grad = True
        phantom_in = phantom_in.cuda()

        depth = 4
        ph_slice = phantom_in[:, :, (2*depth):phantom_in.shape[2] - (2*depth), (2*depth):phantom_in.shape[3] - (2*depth), \
            (2*depth):phantom_in.shape[4] - (2*depth)]
        ph_slice_aux = phantom_in[:, :, depth:phantom_in.shape[2] - depth, depth:phantom_in.shape[3] - depth, \
            depth:phantom_in.shape[4] - depth]
        
        optimiser_main.zero_grad()
        
        #Forward pass
        x, f1= JBFnet(noisy_in, spat_kernel)
        """
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        axes[0].imshow(noisy_in.cpu().detach().numpy()[0, 0, 7], cmap='gray')
        axes[1].imshow(x.cpu().detach().numpy()[0, 0, 0], cmap='gray')
        axes[2].imshow(prior.cpu().detach().numpy()[0, 0, 7], cmap='gray')
        plt.show()
        """
        #Calculate loss
        prim_loss = comb_loss(x, ph_slice)
        aux_loss = comb_loss(f1, ph_slice_aux)
        loss = prim_loss + 0.1 * aux_loss
        
        #Backprop and update
        loss.backward()
        optimiser_main.step()
        
        #print statistics
        running_loss += loss.item()
        #print(time.time() - stime)
        
    JBFnet.eval() 

     #Train step
    for j, data in enumerate(valloader, 0):
        stime = time.time()
        phantom_in, noisy_in = data
        
        #Rescale
        noisy_in = noisy_in/4096
        phantom_in = phantom_in/4096
        
        noisy_in = noisy_in.cuda()
        noisy_in.requires_grad = True
        phantom_in = phantom_in.cuda()

        depth = 4
        ph_slice = phantom_in[:, :, (2*depth):phantom_in.shape[2] - (2*depth), (2*depth):phantom_in.shape[3] - (2*depth), \
            (2*depth):phantom_in.shape[4] - (2*depth)]
        ph_slice_aux = phantom_in[:, :, depth:phantom_in.shape[2] - depth, depth:phantom_in.shape[3] - depth, \
            depth:phantom_in.shape[4] - depth]
        
        #Forward pass
        with torch.no_grad():
            x, f1= JBFnet(noisy_in, spat_kernel)
        
        #Calculate loss     
        prim_loss = comb_loss(x, ph_slice)
        aux_loss = comb_loss(f1, ph_slice_aux)
        loss = prim_loss + 0.1 * aux_loss
        
        #print statistics
        val_loss += loss.item()
        #print(time.time() - stime)

    print('[%d, %5d] train_loss: %f val_loss %f' %
        (epoch + 1, i + 1, running_loss / int(i+1), val_loss / int(j+1) ))
        
    running_loss = 0.0
    val_loss = 0.0
    end = time.time()
    print('Time taken for epoch: '+str(end-start)+' seconds')

    #if (epoch + 1)%5==0:
    print('Saving model...')
    torch.save(JBFnet.state_dict(), 'models/JBFnet_multiscale_'+str(epoch+1)+'.pth')

print('Training complete!')

#%%Load test data
dir = 'C:/Users/z003zv1a/Desktop/Code/reconstucted_ims/Images/Noise comparison/clinical/test/WFBP_25/'
noisy_phantom_test = dicom_read(dir)
dir = 'C:/Users/z003zv1a/Desktop/Code/reconstucted_ims/Images/Noise comparison/clinical/test/WFBP/'
noisy_phantom_gt = dicom_read(dir)
dir = 'C:/Users/z003zv1a/Desktop/Code/reconstucted_ims/Images/Noise comparison/clinical/test/ADMIRE_25/'
noisy_phantom_adm = dicom_read(dir)

# %% SIMPLE TEST
torch.cuda.empty_cache()
indno = 619
denoiser_net = JBF_net().cuda()
denoiser_net.load_state_dict(torch.load('models/JBFnet_multiscale_6.pth'))
denoiser_net.eval()

im = noisy_phantom_test[indno - 8 : indno + 9, :, :]
gt = noisy_phantom_gt[indno, :, :]
adm = noisy_phantom_adm[indno, :, :]
im_cuda = torch.Tensor([im]).cuda()

#Precompute spatial kernel
spat_kernel = torch.zeros(1, 1, 9, 9, 9).cuda()
for a in range(0, 9):
    for b in range(0, 9):
        for c in range(0, 9):
            spat_kernel[0, 0, a, b, c] = torch.Tensor( [distance.euclidean((a, b, c), (4, 4, 4)) ] )

with torch.no_grad():
    dn4, f1 = denoiser_net(im_cuda.view(1, 1, 17, 256, 256)/4096, spat_kernel)

del spat_kernel
del im_cuda
del denoiser_net
torch.cuda.empty_cache()

dn4 = dn4.cpu().detach().numpy()[0,0,0,:,:]*4096
f1 = f1.cpu().detach().numpy()[0,0,4,:,:]*4096
#prior = prior.cpu().detach().numpy()[0,0,7,:,:]*4096
#f2 = f2.cpu().detach().numpy()[0,0,0,:,:]*4096
#f3 = f3.cpu().detach().numpy()[0,0,0,:,:]*4096
#prior = prior.cpu().detach().numpy()[0,0,4,:,:]*4096

title = ['(a)LDCT', '(c)First Block', '(d)JBFnet', '(e)ADMIRE', '(f)SDCT']
im_array = [im[7, :, :],  f1, dn4, adm, gt]

plt.rcParams.update({'font.size': 32})
fig, axes = plt.subplots(1, len(title), figsize=(len(title)*10, 10))
for i, ax in enumerate(axes):
        im_s = im_array[i]
        ax.axis('off')
        ax.imshow(window(im_s), cmap='gray')
        ax.set_title(title[i])
fig.subplots_adjust(hspace=0.1, wspace=0.01)
plt.show()

# %%
