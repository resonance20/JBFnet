# JBFnet - Low Dose CT Denoising by Trainable Joint Bilateral Filtering

This is the reference implementationfor the paper [JBFnet - Low Dose CT Denoising by Trainable Joint Bilateral Filtering](https://link.springer.com/chapter/10.1007/978-3-030-59713-9_49). This implementation consists of the following files:

* **data.py** - Deals with loading data from DICOM and creating a DataLoader object with the training data.
* **helpers.py** - Reads DICOM files into a 16 bit NumPy array.
* **losses.py** - Implements the Edge Filtration loss, and the combined MSE - EF loss.
* **model.py** - Implements the JBFnet architecture, including the JBF blocks.
* **train.py** - Contains the pre-training and main training loops.
* **main.py** - Launches the network training. To launch the code, simply run **main.py** from the command line.

## Requirements
This will probably work with different platforms, but I only tested on Windows 10 with the following settings:

Python 3.7
CUDA 10.1
PyTorch 1.6