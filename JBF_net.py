#%%Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.model_selection import train_test_split
from skimage.measure import compare_ssim
from scipy.spatial import distance
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data
import torch.utils.checkpoint as checkpoint
from torch.autograd import Variable
from pathos.pools import ParallelPool
from EFLoss import EFLoss

from helpers import *

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.backends.cudnn.deterministic=True
torch.manual_seed(1)
torch.cuda.empty_cache()

#%%Load massive train data
ltime = time.time()
dir = 'C:/Users/z003zv1a/Documents/Images/'
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
pool = ParallelPool(8)
noisy_phantom_raw = pool.map(dicom_read, noisy_phantom_list)
phantom_raw = pool.map(dicom_read, phantom_list) 
pool.close()
pool.join()

noisy_phantom = np.concatenate([vol for vol in noisy_phantom_raw], axis=0)
phantom = np.concatenate([np.concatenate([vol, vol, vol, vol], axis=0) for vol in phantom_raw], axis=0)

print('Data loaded!')
print('Time to load: %f seconds'%(round(time.time() - ltime, 2)) )
print(np.shape(phantom))
print(np.shape(noisy_phantom))

#Convert to non-overlapping 3D slabs
print('Converting to slabs...')
thickness = 15
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

#JBFnet specific
phantom = phantom[:, 4:11, :, :]

print(np.shape(phantom))
print(np.shape(noisy_phantom))

#%%Dummy data
#phantom = np.random.randint(0, 4096,size=(80, 7, 64, 64))
#noisy_phantom = np.random.randint(0, 4096,size=(80, 15, 64, 64))

#%%Convert dataset to torch
tensor_phantom = torch.from_numpy(phantom).unsqueeze(1).float()
tensor_noisy = torch.from_numpy(noisy_phantom).unsqueeze(1).float()

dataset = torch.utils.data.TensorDataset(tensor_phantom, tensor_noisy)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
print('Dataset torchified!')

#%%Architecture segments
#Denoiser
class denoiser(nn.Module):
    
    def __init__(self):
        super(denoiser, self).__init__()
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
        #Conv
        x = F.leaky_relu(self.conv1(x))
        im1 = x.clone()[:,:,3:10,:,:]
        x = F.leaky_relu(self.conv2(x))
        im2 = x.clone()[:,:,2:9,:,:]
        x = F.leaky_relu(self.conv3(x))
        im3 = x.clone()[:,:,1:8,:,:]
        x = F.leaky_relu(self.conv4(x))
        #Deconv
        x = F.leaky_relu(self.upconv4(x))
        x = F.leaky_relu(self.pc1(torch.cat((x, im3), 1) ))
        x = F.leaky_relu(self.upconv3(x))
        x = F.leaky_relu(self.pc2(torch.cat((x, im2), 1) ))
        x = F.leaky_relu(self.upconv2(x))
        x = F.leaky_relu(self.pc3(torch.cat((x, im1), 1) ))
        return self.upconv1(x)

#Joint Bilateral Filtering block
class JBF_block(nn.Module):
    
    def __init__(self):
        super(JBF_block, self).__init__()
        self.range_coeffecients = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size = (3, 3, 3), stride = 1),
            nn.ReLU(),
            nn.Conv3d(1, 1, kernel_size = (3, 3, 3), stride = 1),
            nn.ReLU()
        )
        self.domain_coeffecients = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size = (3, 3, 3), stride = 1),
            nn.ReLU(),
            nn.Conv3d(1, 1, kernel_size = (3, 3, 3), stride = 1),
            nn.ReLU()
        )
        
    def forward(self, x, domain_neighbor, guide_im):
        
        #Store shape
        mat_size = (x.shape[0], x.shape[1], x.shape[2] - 4, x.shape[3], x.shape[4])
        
        #Estimate filter coeffecients
        domain_kernel = self.domain_coeffecients(domain_neighbor)
        range_kernel = self.range_coeffecients(guide_im)
        weights = (domain_kernel*range_kernel) + 1e-10
        
        #Apply bilateral filter
        x = F.pad(x, (1, 1, 1, 1, 0, 0), mode='constant')
        x = x.unfold(2, 3, 1).unfold(3, 3, 1).unfold(4, 3, 1).reshape(-1, 1, 3, 3, 3)
        weighted_x = weights*x
        filtered_im = weighted_x.view(weighted_x.shape[0], 1, -1).sum(2) / weights.view(weights.shape[0], 1, -1).sum(2)
        
        #Reshape and upsample
        return filtered_im.view(mat_size)

#%%JBF net architecture
class JBF_net(nn.Module):
    
    def __init__(self):
        super(JBF_net, self).__init__()
        #Denoising
        self.net_denoiser = denoiser()
        self.JBF_block1 = JBF_block()
        self.JBF_block2 = JBF_block()
        self.JBF_block3 = JBF_block()
        self.JBF_block4 = JBF_block()
        
        #Add in parameters
        #self.alfa1 = nn.Parameter(torch.rand(1, 1, 1, 1, 1))
        #self.alfa2 = nn.Parameter(torch.rand(1, 1, 1, 1, 1))
        #self.alfa3 = nn.Parameter(torch.rand(1, 1, 1, 1, 1))
        #self.alfa4 = nn.Parameter(torch.rand(1, 1, 1, 1, 1))
        self.alfa1 = nn.Conv3d(1, 1, kernel_size = (1, 3, 3), stride = 1, padding = (0, 1, 1))
        self.alfa2 = nn.Conv3d(1, 1, kernel_size = (1, 3, 3), stride = 1, padding = (0, 1, 1))
        self.alfa3 = nn.Conv3d(1, 1, kernel_size = (1, 3, 3), stride = 1, padding = (0, 1, 1))
        self.alfa4 = nn.Conv3d(1, 1, kernel_size = (1, 3, 3), stride = 1, padding = (0, 1, 1))
        

    def forward(self, x, spat_kernel):

        #Compute guidance image
        guide_im = self.net_denoiser(x)
        prior = guide_im.clone()
        
        #Compute filter neighborhoods
        guide_im = F.pad(guide_im, (3, 3, 3, 3, 0, 0), mode='constant')
        guide_im = guide_im.unfold(2, 7, 1).unfold(3, 7, 1).unfold(4, 7, 1).reshape(-1, 1, 7, 7, 7)
        guide_im -= guide_im[:, 0, 3, 3, 3].view(guide_im.shape[0], 1, 1, 1, 1)
        guide_im = torch.abs(guide_im)
        
        #Extract relevant part
        inp = x.clone()
        x = x[:, :, 6:9, :, :]
        
        x = F.relu(self.JBF_block1(x, spat_kernel, guide_im))
        x = F.relu( x + self.alfa1( x - inp[:, :, 7:8, :, :]) * ( x - inp[:, :, 7:8, :, :]) )
        f1 = x.clone()
        
        x = torch.cat((inp[:, :, 6:7, :, :], x, inp[:, :, 8:9, :, :]), dim = 2)
        x = F.relu(self.JBF_block1(x, spat_kernel, guide_im))
        x = F.relu( x + self.alfa2( x - inp[:, :, 7:8, :, :]) * ( x - inp[:, :, 7:8, :, :]) )
        f2 = x.clone()

        x = torch.cat((inp[:, :, 6:7, :, :], x, inp[:, :, 8:9, :, :]), dim = 2)
        x = F.relu(self.JBF_block1(x, spat_kernel, guide_im))
        x = F.relu( x + self.alfa3( x - inp[:, :, 7:8, :, :]) * ( x - inp[:, :, 7:8, :, :]) )
        f3 = x.clone()

        x = torch.cat((inp[:, :, 6:7, :, :], x, inp[:, :, 8:9, :, :]), dim = 2)
        x = F.relu(self.JBF_block1(x, spat_kernel, guide_im))
        x = F.relu( x + self.alfa4( x - inp[:, :, 7:8, :, :]) * ( x - inp[:, :, 7:8, :, :]) )

        return x, prior, f1, f2, f3

#%%Loss functions and optimiser
try:
    del JBFnet
    torch.cuda.empty_cache()
except NameError:
    print('Tensors not initialised')
JBFnet = JBF_net().cuda()
#JBFnet.load_state_dict(torch.load('models/JBFnet_pretrain_10.pth'))

den_params = list(filter(lambda p: 'denoiser' in p[0], JBFnet.named_parameters()))
den_params = [p[1] for p in den_params]
#net_params = list(filter(lambda p: 'denoiser' not in p[0], JBFnet.named_parameters()))
#net_params = [p[1] for p in net_params]
optimiser_main = optim.Adam(JBFnet.parameters(), lr = 1e-4)
optimiser_prior = optim.Adam(den_params, lr = 1e-4)

model_parameters = filter(lambda p: p.requires_grad, JBFnet.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print('Number of trainable parameters: ' +str(params))

#%%Pre train prior

for epoch in range(10):
    running_loss = 0

    #Precompute spatial kernel
    spat_kernel = torch.zeros(1, 1, 7, 7, 7).cuda()
    for a in range(0, 7):
        for b in range(0, 7):
            for c in range(0, 7):
                spat_kernel[0, 0, a, b, c] = torch.Tensor( [distance.euclidean((a, b, c), (3, 3, 3)) ] )

    JBFnet.train() 
    start = time.time()

     #Train step
    for i, data in enumerate(dataloader, 0):
        
        phantom_in, noisy_in = data
        
        #Rescale
        noisy_in = noisy_in/4096
        phantom_in = phantom_in/4096
        
        noisy_in = noisy_in.cuda()
        noisy_in.requires_grad = True
        phantom_in = phantom_in.cuda()
        
        optimiser_prior.zero_grad()
        
        #Forward pass
        _, prior, _, _, _ = JBFnet(noisy_in, spat_kernel)
        
        #Calculate loss
        loss = nn.MSELoss()(prior, phantom_in)
        
        #Backprop and update
        loss.backward()
        optimiser_prior.step()
        
        #print statistics
        running_loss += loss.item()
        
    print('[%d, %5d] train_loss: %f' %
        (epoch + 1, i + 1, running_loss / int(i+1) ))
        
    running_loss = 0.0
    end = time.time()
    print('Time taken for epoch: '+str(end-start)+' seconds')

    if (epoch + 1)%5==0:
        print('Saving model...')
        torch.save(JBFnet.state_dict(), 'models/JBFnet_pretrain_'+str(epoch+1)+'.pth')

print('Pretraining complete!')

#%%Train branch
def comb_loss(im1, im2):
    return nn.MSELoss()(im1, im2) + \
        0.1 * EFLoss()(im1.view(-1, 1, im1.shape[3], im1.shape[4]), im2.view(-1, 1, im2.shape[3], im2.shape[4]))

for epoch in range(10, 30):
    running_loss = 0
    val_loss = 0

    #Precompute spatial kernel
    spat_kernel = torch.zeros(1, 1, 7, 7, 7).cuda()
    for a in range(0, 7):
        for b in range(0, 7):
            for c in range(0, 7):
                spat_kernel[0, 0, a, b, c] = torch.Tensor( [distance.euclidean((a, b, c), (3, 3, 3)) ] )

    JBFnet.train() 
    start = time.time()

     #Train step
    for i, data in enumerate(dataloader, 0):
        
        #Load data and initialise optimisers
        phantom_in, noisy_in = data
        
        #Rescale
        noisy_in = noisy_in/4096
        phantom_in = phantom_in/4096
        
        phantom_in = phantom_in.cuda()
        ph_slice = phantom_in[:, :, 4:5, :, :]
        noisy_in = noisy_in.cuda()
        noisy_in.requires_grad=True

        optimiser_main.zero_grad()
        
        #Forward pass
        im4, prior, f1, f2, f3 = JBFnet(noisy_in, spat_kernel)
       
        #Calculate deep supervision loss
        prim_loss = comb_loss(im4, ph_slice)
        prior_loss = nn.MSELoss()(prior, phantom_in)
        aux_loss = comb_loss(f1, ph_slice) + comb_loss(f2, ph_slice) + comb_loss(f3, ph_slice)

        loss = prim_loss + 0.1 * prior_loss + 0.1 * aux_loss
        loss.backward()
        optimiser_main.step()

        # print statistics
        running_loss += loss.item()

    print('[%d, %5d] train_loss: %f ' %
            (epoch + 1, i + 1, running_loss / int(i+1)))
    running_loss = 0.0
    val_loss = 0.0
    end = time.time()
    print('Time taken for epoch: '+str(end-start)+' seconds')

    if (epoch + 1)%5==0:
        print('Saving model...')
        torch.save(JBFnet.state_dict(), 'models/JBFnet_'+str(epoch+1)+'.pth')

print('Training complete!')

#%%Load test data
dir = 'C:/Users/z003zv1a/Desktop/Code/reconstucted_ims/Images/Noise comparison/clinical/test/WFBP/'
noisy_phantom_test = dicom_read(dir)
dir = 'C:/Users/z003zv1a/Desktop/Code/reconstucted_ims/Images/Noise comparison/clinical/test/WFBP/'
noisy_phantom_gt = dicom_read(dir)
dir = 'C:/Users/z003zv1a/Desktop/Code/reconstucted_ims/Images/Noise comparison/clinical/test/ADMIRE/'
noisy_phantom_adm = dicom_read(dir)

# %% SIMPLE TEST
indno = 225
denoiser_net = JBF_net().cuda()
denoiser_net.load_state_dict(torch.load('models/JBFnet_30.pth'))
denoiser_net.eval()

im = noisy_phantom_test[indno - 7 : indno + 8, :, :]
gt = noisy_phantom_gt[indno, :, :]
adm = noisy_phantom_adm[indno, :, :]
im_cuda = torch.Tensor([im]).cuda()

#Precompute spatial kernel
spat_kernel = torch.zeros(1, 1, 7, 7, 7).cuda()
for a in range(0, 7):
    for b in range(0, 7):
        for c in range(0, 7):
            spat_kernel[0, 0, a, b, c] = torch.Tensor( [distance.euclidean((a, b, c), (3, 3, 3)) ] )

with torch.no_grad():
    dn4, prior, f1, f2, f3 = denoiser_net(im_cuda.view(1, 1, 15, 256, 256)/4096, spat_kernel)

del spat_kernel
del im_cuda
del denoiser_net
torch.cuda.empty_cache()

dn4 = dn4.cpu().detach().numpy()[0,0,0,:,:]*4096
f1 = f1.cpu().detach().numpy()[0,0,0,:,:]*4096
f2 = f2.cpu().detach().numpy()[0,0,0,:,:]*4096
f3 = f3.cpu().detach().numpy()[0,0,0,:,:]*4096
prior = prior.cpu().detach().numpy()[0,0,4,:,:]*4096

title = ['(a)SDCT', '(b)JBFnet', '(c)ADMIRE']
im_array = [im[7, :, :], dn4, adm]

plt.rcParams.update({'font.size': 32})
fig, axes = plt.subplots(1, 3, figsize=(30, 10))
for i, ax in enumerate(axes):
        im_s = im_array[i]
        ax.axis('off')
        ax.imshow(window(im_s), cmap='gray')
        ax.set_title(title[i])
fig.subplots_adjust(hspace=0.1, wspace=0.01)
plt.show()
"""
fig, axes = plt.subplots(2, 4 , figsize=(40,20))
for axe in axes:
    for ax in axe:
        ax.axis('off')
axes[0, 0].imshow(window(im[7, :, :]), cmap='gray')
axes[0, 1].imshow(window(prior, slope=4096), cmap='gray')
axes[0, 2].imshow(window(dn4, slope=4096), cmap='gray')
axes[0, 3].imshow(window(gt[7, :, :]), cmap='gray')
axes[1, 0].imshow(window(f1, slope=4096), cmap='gray')
axes[1, 1].imshow(window(f2, slope=4096), cmap='gray')
axes[1, 2].imshow(window(f3, slope=4096), cmap='gray')
axes[1, 3].imshow(np.ones((256,256)), cmap='gray')
fig.subplots_adjust(hspace=0.01, wspace=0.01)
plt.show()


print('Output')
print(psnr(f1*4096, gt[7, :, :]))
print(compare_ssim(dn4*4096, gt[7, :, :], data_range=4096))

print('Input')
print(psnr(im[7, :, :], gt[7, :, :]))
print(compare_ssim(im[7, :, :], gt[7, :, :], data_range=4096))

print('Prior')
print(psnr(prior*4096, gt[7, :, :]))
print(compare_ssim(prior*4096, gt[7, :, :], data_range=4096))
"""

# %%
