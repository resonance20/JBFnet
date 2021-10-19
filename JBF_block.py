#Imports
import torch.nn as nn
import torch.nn.functional as F
from math import floor

#Joint Bilateral Filtering block
class JBF_block(nn.Module):
    
    def __init__(self, n_layers=2, bil_filt_size=3):
        
        assert n_layers > 0
        assert bil_filt_size > 0

        super(JBF_block, self).__init__()
        self.n_layers = n_layers
        self.bil_filt_size = bil_filt_size

        self.range_coeffecients = []
        self.domain_coeffecients = []

        for layer in range(n_layers):
            self.range_coeffecients.append(nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=1))
            self.range_coeffecients.append(nn.ReLU())
            self.domain_coeffecients.append(nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=1))
            self.domain_coeffecients.append(nn.ReLU())

        self.range_coeffecients = nn.Sequential(*self.range_coeffecients)
        self.domain_coeffecients = nn.Sequential(*self.domain_coeffecients)
        
    def forward(self, x, domain_neighbor, guide_im):
        
        assert (domain_neighbor.shape[2] - self.n_layers*2) >= self.bil_filt_size
        for dim in [2, 3, 4]:
            assert guide_im.shape[dim] == domain_neighbor.shape[dim]

        #Store shape
        mat_size = (x.shape[0], x.shape[1], x.shape[2] - self.n_layers*2, x.shape[3], x.shape[4])
        
        #Estimate filter coeffecients
        domain_kernel = self.domain_coeffecients(domain_neighbor)
        range_kernel = self.range_coeffecients(guide_im)
        weights = (domain_kernel*range_kernel) + 1e-10
        
        #Apply bilateral filter
        pad_depth = floor(self.bil_filt_size/2)

        x = F.pad(x, (pad_depth, pad_depth, pad_depth, pad_depth, 0, 0), mode='constant')#We don't pad in the z-direction
        x = x.unfold(2, self.bil_filt_size, 1).unfold(3, self.bil_filt_size, 1).unfold(4, self.bil_filt_size, 1)\
            .reshape(-1, 1, self.bil_filt_size, self.bil_filt_size, self.bil_filt_size)
        weighted_x = weights*x

        filtered_im = weighted_x.view(weighted_x.shape[0], 1, -1).sum(2) / weights.view(weights.shape[0], 1, -1).sum(2)
        
        #Reshape and upsample
        return filtered_im.view(mat_size)

##Please add the code for the Inception-JBF block here