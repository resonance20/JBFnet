import pydicom
import os
import numpy as np
import cv2

def dicom_read(path):

    #Load list of dicom files
    list_files = os.listdir(path)
    list_dicom = []
    for file in list_files:
        if file.endswith('.dcm') or file.endswith('.IMA'):
            list_dicom.append(file)

    #Find reference values
    RefDs = pydicom.read_file(path + list_dicom[0])
    #const_pixel_dims = (len(list_dicom), RefDs.Rows, RefDs.Columns, )
    const_pixel_dims = (len(list_dicom), 256, 256)
    
    #Create array and load values
    dicom_array = np.zeros(const_pixel_dims)
    for file in list_dicom:
        ds = pydicom.dcmread(path + file)
        im = np.array(ds.pixel_array, np.int16)
        dicom_array[list_dicom.index(file),:,:] = cv2.resize(im, (256, 256))
    
    return dicom_array