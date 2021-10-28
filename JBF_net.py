#%%Imports
from scipy.spatial import distance

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor

from prior import DL_prior
from JBF_block import JBF_block

#%%JBF net architecture
class JBF_net(nn.Module):
    
    def __init__(self, kernel_size=7, bil_filt_size=3, num_blocks=4):

        """! Class implementation of JBFnet
        @param kernel_size  Receptive field size of the kernel needed for estimating the filter functions. Should be an odd number, and reaches the desired receptive field by stacking 3 x 3 layers.
        @param bil_filt_size  Size of the 3D bilateral filter which is calculated in the JBF block. Should be an odd number atleast 2 smaller than kernel_size.
        @param num_blocks   Number of JBF blocks needed. Atleast 1 block is necessary.
        @return  JBFnet object
        """


        assert kernel_size > bil_filt_size
        assert num_blocks > 0

        super(JBF_net, self).__init__()

        #Denoising
        n_layers = int((kernel_size - bil_filt_size)/2)
        self.net_denoiser = DL_prior(32)
        self.blocks = []
        self.alfas = []
        for block in num_blocks:
            self.blocks.append( JBF_block(n_layers=n_layers, bil_filt_size=bil_filt_size) )
            self.alfas.append( nn.Conv3d(1, 1, kernel_size = (1, 3, 3), stride = 1, padding = (0, 1, 1)) )

        #Distance matrix
        center = floor(kernel_size/2)
        self.spat_kernel = torch.zeros(1, 1, kernel_size, kernel_size, kernel_size).cuda()
        for a in range(0, kernel_size):
            for b in range(0, kernel_size):
                for c in range(0, kernel_size):
                    self.spat_kernel[0, 0, a, b, c] = torch.Tensor( [distance.euclidean((a, b, c), (center, center, center)) ] )

        #Other params
        self.bil_filt_size = bil_filt_size
        self.kernel_size = kernel_size
        

    def forward(self, x):

        assert x.shape[2] > (self.bil_filt_size + 2)

        #Compute guidance image
        guide_im = self.net_denoiser(x)
        prior = guide_im.clone()
        int_results = []
        
        #Compute filter neighborhoods
        pad_depth = floor(self.kernel_size/2)

        guide_im = F.pad(guide_im, (pad_depth, pad_depth, pad_depth, pad_depth, 0, 0), mode='constant')
        guide_im = guide_im.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1).unfold(4, self.kernel_size, 1)\
            .reshape(-1, 1, self.kernel_size, self.kernel_size, self.kernel_size)
        guide_im -= guide_im[:, 0, pad_depth, pad_depth, pad_depth].view(guide_im.shape[0], 1, 1, 1, 1)
        guide_im = torch.abs(guide_im)
        
        #Extract relevant part
        inp = x.clone()
        centre_shape = floor(x.shape[2]/2) 
        x = x[:, :, centre_shape - floor(self.bil_filt_size/2):-(centre_shape - floor(self.bil_filt_size/2))]
        
        for i in range(0, len(self.blocks)):
            
            x = F.relu(self.blocks[i](x, self.spat_kernel, guide_im))
            x = F.relu( x + self.alfas[i]( x - inp[:, :, centre_shape:-centre_shape]) * ( x - inp[:, :, centre_shape:-centre_shape]) )
            int_results.append(x.clone())
            if i != (len(self.blocks) - 1):
                x = F.relu(torch.cat((inp[:, :, centre_shape - floor(self.bil_filt_size/2):centre_shape], x, \
                    inp[:, :, centre_shape + floor(self.bil_filt_size/2):-(centre_shape - floor(self.bil_filt_size/2))]), dim = 2))

        return x, prior, int_results[:-1]

