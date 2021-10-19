#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#CPCE3D Prior architecture
class DL_prior(nn.Module):
    
    def __init__(self, n_filters=32):
        super(DL_prior, self).__init__()
        self.conv1 = nn.Conv3d(1, n_filters, kernel_size = (3, 3, 3), stride = 1)
        self.conv2 = nn.Conv3d(n_filters, n_filters, kernel_size = (3, 3, 3), stride = 1)
        self.conv3 = nn.Conv3d(n_filters, n_filters, kernel_size = (3, 3, 3), stride = 1)
        self.conv4 = nn.Conv3d(n_filters, n_filters, kernel_size = (3, 3, 3), stride = 1)
        self.pc1 = nn.Conv3d(2*n_filters, n_filters, kernel_size = (1, 1, 1), stride = 1)
        self.pc2 = nn.Conv3d(2*n_filters, n_filters, kernel_size = (1, 1, 1), stride = 1)
        self.pc3 = nn.Conv3d(2*n_filters, n_filters, kernel_size = (1, 1, 1), stride = 1)
        self.upconv4 = nn.ConvTranspose3d(n_filters, n_filters, kernel_size = (1, 3, 3), stride = 1)
        self.upconv3 = nn.ConvTranspose3d(n_filters, n_filters, kernel_size = (1, 3, 3), stride = 1)
        self.upconv2 = nn.ConvTranspose3d(n_filters, n_filters, kernel_size = (1, 3, 3), stride = 1)
        self.upconv1 = nn.ConvTranspose3d(n_filters, 1, kernel_size = (1, 3, 3), stride = 1)

    def forward(self, x):

        assert x.shape[2] >= 9

        #Conv
        x = F.leaky_relu(self.conv1(x))
        im1 = x.clone()[:,:,3:-3,:,:]
        x = F.leaky_relu(self.conv2(x))
        im2 = x.clone()[:,:,2:-2,:,:]
        x = F.leaky_relu(self.conv3(x))
        im3 = x.clone()[:,:,1:-1,:,:]
        x = F.leaky_relu(self.conv4(x))
        #Deconv
        x = F.leaky_relu(self.upconv4(x))
        x = F.leaky_relu(self.pc1(torch.cat((x, im3), 1) ))
        x = F.leaky_relu(self.upconv3(x))
        x = F.leaky_relu(self.pc2(torch.cat((x, im2), 1) ))
        x = F.leaky_relu(self.upconv2(x))
        x = F.leaky_relu(self.pc3(torch.cat((x, im1), 1) ))
        return self.upconv1(x)

##Please add the code for the NL means prior here