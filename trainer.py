#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from EFLoss import EFLoss
import numpy as np
from time import time

from JBF_net import JBF_net

class JBFnet_trainer():

    def __init__(self, kernel_size=7, bil_filt_size=3, num_blocks=4):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = JBF_net(kernel_size=kernel_size, bil_filt_size=bil_filt_size, num_blocks=num_blocks).to(device)
        self.net.spat_kernel = self.net.spat_kernel.to(device)

        den_params = list(filter(lambda p: 'denoiser' in p[0], self.net.named_parameters()))
        den_params = [p[1] for p in den_params]
        self.optimiser_main = optim.Adam(self.net.parameters(), lr = 1e-4)
        self.optimiser_prior = optim.Adam(den_params, lr = 1e-4)

        model_parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        print('Number of trainable parameters: ' +str(params))

    #Pre train prior image #TODO:Add validation loss
    def pre_train(self, train_data, epoch_number=10):

        for epoch in range(epoch_number):
            running_loss = 0

            self.net.train() 
            start = time.time()

            #Train step
            for i, data in enumerate(train_data, 0):
                
                phantom_in, noisy_in = data
                
                #Rescale
                noisy_in = noisy_in/4096
                phantom_in = phantom_in/4096
                
                noisy_in = noisy_in.cuda()
                noisy_in.requires_grad = True
                phantom_in = phantom_in.cuda()
                
                self.optimiser_prior.zero_grad()
                
                #Forward pass
                _, prior, _= self.net(noisy_in)
                shrinkage = int((noisy_in.shape[2] - prior.shape[2])/2)
                
                #Calculate loss
                loss = nn.MSELoss()(prior, phantom_in[:, : shrinkage:-shrinkage])
                
                #Backprop and update
                loss.backward()
                self.optimiser_prior.step()
                
                #print statistics
                running_loss += loss.item()
                
            print('[%d, %5d] train_loss: %f' %
                (epoch + 1, i + 1, running_loss / int(i+1) ))
                
            running_loss = 0.0
            end = time.time()
            print('Time taken for epoch: '+str(end-start)+' seconds')

            if (epoch + 1)%5==0:
                print('Saving model...')
                torch.save(self.net.state_dict(), 'models/JBFnet_pretrain_'+str(epoch+1)+'.pth')

        print('Pretraining complete!')

    #Loss function for edge preservation
    def comb_loss(self, im1, im2):
        return nn.MSELoss()(im1, im2) + \
            0.1 * EFLoss()(im1.view(-1, 1, im1.shape[3], im1.shape[4]), im2.view(-1, 1, im2.shape[3], im2.shape[4]))

    #Main train loop #TODO:Add validation loss
    def train(self, train_data, epoch_number=20):

        for epoch in range(epoch_number):
            running_loss = 0

            self.net.train() 
            start = time.time()

            #Train step
            for i, data in enumerate(train_data, 0):
                
                #Load data and initialise optimisers
                phantom_in, noisy_in = data
                
                #Rescale
                noisy_in = noisy_in/4096
                phantom_in = phantom_in/4096
                
                phantom_in = phantom_in.cuda()
                noisy_in = noisy_in.cuda()
                noisy_in.requires_grad=True

                self.optimiser_main.zero_grad()
                
                #Forward pass
                im4, prior, int_result = self.net(noisy_in)
                shrinkage_prior = int((noisy_in.shape[2] - prior.shape[2])/2)
                shrinkage_output = int((noisy_in.shape[2] - im4.shape[2])/2)
            
                #Calculate deep supervision loss
                prim_loss = self.comb_loss(im4, phantom_in[:, : shrinkage_output:-shrinkage_output])
                prior_loss = nn.MSELoss()(prior, phantom_in[:, : shrinkage_prior:-shrinkage_prior])
                aux_loss = 0
                
                for f in int_result:
                    aux_loss += self.comb_loss(f, phantom_in[:, : shrinkage_output:-shrinkage_output])

                loss = prim_loss + 0.1 * prior_loss + 0.1 * aux_loss
                loss.backward()
                self.optimiser_main.step()

                # print statistics
                running_loss += loss.item()

            print('[%d, %5d] train_loss: %f ' %
                    (epoch + 1, i + 1, running_loss / int(i+1)))
            running_loss = 0.0

            end = time.time()
            print('Time taken for epoch: '+str(end-start)+' seconds')

            if (epoch + 1)%5==0:
                print('Saving model...')
                torch.save(self.net.state_dict(), 'models/JBFnet_'+str(epoch+1)+'.pth')

        print('Training complete!')