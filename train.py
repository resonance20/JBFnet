import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from scipy.spatial import distance

from model import JBF_net
from losses import comb_loss

#class to train model
class train_jbfnet():

    def __init__(self):
        super(train_jbfnet, self).__init__()
        self.optimiser_main = None
        self.optimiser_prior = None
        self.model = None

    #Loss functions and optimiser
    def initialise_models(self):
        
        try:
            del self.model
            torch.cuda.empty_cache()
        except NameError:
            print('Tensors not initialised')
        self.model = JBF_net().cuda()

        #Separate optimisers for pre training and normal training
        den_params = list(filter(lambda p: 'denoiser' in p[0], self.model.named_parameters()))
        den_params = [p[1] for p in den_params]
        self.optimiser_main = optim.Adam(self.model.parameters(), lr = 1e-4)
        self.optimiser_prior = optim.Adam(den_params, lr = 1e-4)

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        print('Number of trainable parameters: ' +str(params))

    #Pre train prior
    def pre_train(self, dataloader, epoch_range=10):

        for epoch in range(epoch_range):
            running_loss = 0

            #Precompute spatial kernel
            spat_kernel = torch.zeros(1, 1, 7, 7, 7).cuda()
            for a in range(0, 7):
                for b in range(0, 7):
                    for c in range(0, 7):
                        spat_kernel[0, 0, a, b, c] = torch.Tensor( [distance.euclidean((a, b, c), (3, 3, 3)) ] )

            self.model.train() 
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
                
                self.optimiser_prior.zero_grad()
                
                #Forward pass
                _, prior, _, _, _ = self.model(noisy_in, spat_kernel)
                
                #Calculate loss
                loss = nn.MSELoss()(prior, phantom_in)
                
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
                torch.save(self.model.state_dict(), 'models/JBFnet_pretrain_'+str(epoch+1)+'.pth')

        print('Pretraining complete!')

    #Main train loop
    def main_train_loop(self, dataloader, epoch_range=20):

        self.model.load_state_dict(torch.load('models/JBFnet_pretrain_20.pth'))

        for epoch in range(epoch_range):
            running_loss = 0

            #Precompute spatial kernel
            spat_kernel = torch.zeros(1, 1, 7, 7, 7).cuda()
            for a in range(0, 7):
                for b in range(0, 7):
                    for c in range(0, 7):
                        spat_kernel[0, 0, a, b, c] = torch.Tensor( [distance.euclidean((a, b, c), (3, 3, 3)) ] )

            self.model.train() 
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

                self.optimiser_main.zero_grad()
                
                #Forward pass
                im4, prior, f1, f2, f3 = self.model(noisy_in, spat_kernel)
            
                #Calculate deep supervision loss
                loss_unit = comb_loss()
                prim_loss = loss_unit(im4, ph_slice)
                prior_loss = nn.MSELoss()(prior, phantom_in)
                aux_loss = loss_unit(f1, ph_slice) + loss_unit(f2, ph_slice) + loss_unit(f3, ph_slice)

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
                torch.save(self.model.state_dict(), 'models/JBFnet_'+str(epoch+1)+'.pth')

        print('Training complete!')

    #Main class function
    def train_model(self, dataloader, epoch_range_pt=10, epoch_range=20):

        self.initialise_models()
        self.pre_train(dataloader=dataloader, epoch_range=epoch_range_pt)
        self.main_train_loop(dataloader=dataloader, epoch_range=epoch_range)