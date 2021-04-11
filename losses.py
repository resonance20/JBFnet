import torch
import torch.nn as nn
import torch.nn.functional as F

#Edge filtration loss class
class EFLoss(nn.Module):

    def __init__(self):
        super(EFLoss, self).__init__()

    #Sobel filter in 3D
    def gradIm3d(self, image):

        if image.is_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        filtx = torch.Tensor([
            [[-1, 0, 1], [0, 0, 2], [-2, 0, 1]], 
            [[-2, 0, -2], [-4, 0, 4], [-2, 0, 2]], 
            [[-1, 0, 1], [0, 0, 2], [-2, 0, 1]]
            ]).to(device)

        filty = torch.Tensor([
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
            [[2, 4, 2], [0, 0, 0], [-2, -4, -2]], 
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
            ]).to(device)

        filtz = torch.Tensor([
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], 
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
            ]).to(device)

        Gx = F.conv3d(image, filtx.view((1,1,3,3,3)))
        Gy = F.conv3d(image, filty.view((1,1,3,3,3)))
        Gz = F.conv3d(image, filtz.view((1,1,3,3,3)))

        return Gx, Gy, Gz

    #Sobel filter in 2D
    def gradIm2d(self, image):

        if image.is_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    
        filtx = torch.Tensor([[1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]]).to(device)

        filty = torch.Tensor([[1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]]).to(device)

        Gx = F.conv2d(image, filtx.view((1,1,3,3)))
        Gy = F.conv2d(image, filty.view((1,1,3,3)))

        return Gx, Gy

    def loss2d(self, im1, im2):

        Gx1, Gy1 = self.gradIm2d(im1)
        Gx2, Gy2 = self.gradIm2d(im2)

        loss_x = torch.abs(Gx1 - Gx2)
        loss_y = torch.abs(Gy1 - Gy2)

        return torch.mean(loss_x + loss_y)

    def loss3d(self, im1, im2):

        Gx1, Gy1, Gz1 = self.gradIm3d(im1)
        Gx2, Gy2, Gz2 = self.gradIm3d(im2)

        loss_x = torch.abs(Gx1 - Gx2)
        loss_y = torch.abs(Gy1 - Gy2)
        loss_z = torch.abs(Gz1 - Gz2)

        return torch.mean(loss_x + loss_y + loss_z)

    def forward(self, im1, im2):

        if im1.shape != im2.shape:
            raise Exception('Input shapes do not match!!')

        if len(list(im1.shape)) == 4:
            return self.loss2d(im1, im2)

        elif len(list(im1.shape)) == 5:
            if(im1.shape[2]) == 1:
                return self.loss2d(im1.squeeze(2), im2.squeeze(2))
            else:
                return self.loss3d(im1, im2)

        else:
            raise Exception('Unsupported number of dimensions!!')

#Combination of MSE loss and EF loss
class comb_loss(nn.Module):

    def __init__(self):
        super(comb_loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ef_loss = EFLoss()
        self.lamb = 0.1

    def foward(self, im1, im2):
        return self.mse_loss(im1, im2) + self.lamb * self.ef_loss(im1, im2)
