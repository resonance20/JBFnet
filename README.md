# JBFnet

An implementation of JBFnet as first presented by [Patwari et al.](https://arxiv.org/abs/2007.04754) at MICCAI 2020.

## Explanation
This contains an extended version of the original parameter. The JBFnet has some extra parameters, which are user defined. The class description is given in ```JBF_net.py```.

```python
class JBF_net(nn.Module):
    
    def __init__(self, kernel_size=7, bil_filt_size=3, num_blocks=4):

        """! Class implementation of JBFnet
        @param kernel_size  Receptive field size of the kernel needed for estimating the filter functions. Should be an odd number, and reaches the desired receptive field by stacking 3 x 3 layers.
        @param bil_filt_size  Size of the 3D bilateral filter which is calculated in the JBF block. Should be an odd number atleast 2 smaller than kernel_size.
        @param num_blocks   Number of JBF blocks needed. Atleast 1 block is necessary.
        @return  JBFnet object
        """
```

Functions to train are provided in ```trainer.py```. However, they have not been tested and will not be described.

Inference has not been implemented yet. It should however follow the format described in the DLDenoise repo.