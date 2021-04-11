#Imports
import torch

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.backends.cudnn.deterministic=True
torch.manual_seed(1)
torch.cuda.empty_cache()

from train import train_jbfnet
from data import data_prep

#Main
if __name__ == "__main__":
    data_handler = data_prep()
    dataloader = data_handler.create_dataloader()
    train_handler = train_jbfnet()
    train_handler.train_model(dataloader=dataloader)