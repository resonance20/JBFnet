import os
from pathos.pools import ParallelPool
import numpy as np
import time
from helpers import dicom_read

import torch

#Load massive train data
class data_prep():

    def __init__(self):
        super(data_prep).__init__()
        self.noisy_phantom = []
        self.phantom = []

    #Loading and storing data in a numpy array
    def load_data(self, num_patients=10, num_threads=8):
        ltime = time.time()
        dir = 'C:/Users/z003zv1a/Documents/Images/Old/'

        phantom_list = []
        noisy_phantom_list = []

        print('Loading data...')

        for no, patient in enumerate(os.listdir(dir)):

            #We limit it to ten patients to begin with
            if no>=num_patients:
                break

            newdir = dir + patient
            if os.path.isdir(newdir):

                for recon in os.listdir(newdir):
                    newdir2 = os.path.join(newdir, recon)
                    if os.path.isdir(newdir2) and 'WFBP' in recon:

                        for doselevel in os.listdir(newdir2):
                            finaldir = os.path.join(newdir2, doselevel)
                            if (os.path.isdir(finaldir)) and '100' in doselevel:
                                phantom_list.append(finaldir + '/')

                            if (os.path.isdir(finaldir)) and '100' not in doselevel and 'High' not in doselevel:
                                noisy_phantom_list.append(finaldir + '/')

        pool = ParallelPool(num_threads)
        noisy_phantom_raw = pool.map(dicom_read, noisy_phantom_list)
        phantom_raw = pool.map(dicom_read, phantom_list) 
        pool.close()
        pool.join()
        pool.clear()

        self.noisy_phantom = np.concatenate([vol for vol in noisy_phantom_raw], axis=0)
        self.phantom = np.concatenate([np.concatenate([vol, vol, vol, vol], axis=0) for vol in phantom_raw], axis=0)

        print('Data loaded!')
        print('Time to load: %f seconds'%(round(time.time() - ltime, 2)) )
        print(np.shape(self.phantom))
        print(np.shape(self.noisy_phantom))

    #Converting to patches
    def prepare_slabs(self, thickness=15, psize=64):

        #Convert to non-overlapping 3D slabs
        print('Converting to slabs...')
        if self.phantom.shape[0]%thickness is not 0:
            padding_depth = thickness - self.phantom.shape[0]%thickness
            self.phantom = np.concatenate((self.phantom, np.zeros([padding_depth, self.phantom.shape[1], self.phantom.shape[2]])), axis = 0)
            self.noisy_phantom = np.concatenate((self.noisy_phantom, np.zeros([padding_depth, self.noisy_phantom.shape[1], self.noisy_phantom.shape[2]])), axis = 0)
        self.phantom = np.reshape(self.phantom, (-1, thickness, self.phantom.shape[1], self.phantom.shape[2]))
        self.noisy_phantom = np.reshape(self.noisy_phantom, (-1, thickness, self.noisy_phantom.shape[1], self.noisy_phantom.shape[2]))

        print(np.shape(self.phantom))
        print(np.shape(self.noisy_phantom))

        #Convert to 3D patches
        print('Converting to patches...')
        num_patches = int(self.phantom.shape[2]/psize)
        self.phantom = self.phantom.reshape(self.phantom.shape[0], thickness, num_patches, psize, num_patches, psize).swapaxes(3, 4). \
        reshape(self.phantom.shape[0], thickness, -1, psize, psize).swapaxes(1, 2).reshape(-1, thickness, psize, psize)
        self.noisy_phantom = self.noisy_phantom.reshape(self.noisy_phantom.shape[0], thickness, num_patches, psize, num_patches, psize).swapaxes(3, 4). \
        reshape(self.noisy_phantom.shape[0], thickness, -1, psize, psize).swapaxes(1, 2).reshape(-1, thickness, psize, psize)

        #JBFnet specific
        self.phantom = self.phantom[:, 4:11, :, :]

        print(np.shape(self.phantom))
        print(np.shape(self.noisy_phantom))

    #Convert dataset to torch
    def torchify(self):
        tensor_phantom = torch.from_numpy(self.phantom).unsqueeze(1).float()
        tensor_noisy = torch.from_numpy(self.noisy_phantom).unsqueeze(1).float()

        dataset = torch.utils.data.TensorDataset(tensor_phantom, tensor_noisy)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        print('Dataset torchified!')
        return dataloader

    #Main class function
    def create_dataloader(self, num_patients=10, num_threads=16, thickness=15, psize=64):

        self.load_data(num_patients=num_patients, num_threads=num_threads)
        self.prepare_slabs(thickness=thickness, psize=psize)
        return self.torchify()