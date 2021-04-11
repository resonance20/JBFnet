import torch
import torch.nn as nn
import torch.nn.functional as F

#Prior estimator
class prior_est(nn.Module):
    
    def __init__(self):
        super(prior_est, self).__init__()
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

#JBF net architecture
class JBF_net(nn.Module):
    
    def __init__(self):
        super(JBF_net, self).__init__()
        #Denoising
        self.net_denoiser = prior_est()
        self.JBF_block1 = JBF_block()
        self.JBF_block2 = JBF_block()
        self.JBF_block3 = JBF_block()
        self.JBF_block4 = JBF_block()
        
        #Add in parameters
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