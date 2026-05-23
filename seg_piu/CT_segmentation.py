import os
import numpy as np
import pydicom
import torch

from residualUnet import *
from metrics import dice_coef, loss_dice
#from preprocessingFunctions import *
from segmentation_functions import * 
from AuxOpen_dicom import load_DICOM, load_PT

from keras.optimizers import Adam
import matplotlib.pyplot as plt
import re
 

#-----------RESIDUAL UNET-----------
# Define loss function
criterion_dice = loss_dice
    
# Define optimizer
opt = Adam(learning_rate=1e-4)
    
# We now compile the model
resUnet = residualUNet()
resUnet.compile(optimizer=opt, loss = criterion_dice, metrics=[dice_coef])

# Path to where weights are stored
checkpoint_path = "bestmodel.epoch05.hdf5"

# Load weights to model
resUnet.load_weights(checkpoint_path)


#-------------CT DATA LOAD EXAMPLE-------------
#data_path = "X:/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/dicoms/LIDC-IDRI-0001/01-01-2000-30178/3000566.000000-03192/"
#data_path = "X:/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/dicoms/LIDC-IDRI-0032/01-01-2000-53482/3000537.000000-91689/"
#scan = load_DICOM(data_path)

"""
data_path = "C:/Users/pedro.fernandes.sous/Desktop/Sandbox/hu_generation_samples/sample_0.pt"
#data_path = "C:/Users/pedro.fernandes.sous/Desktop/Pilot/pilot_segmentation/main/ct.pt"
scan = load_PT(data_path)
scan_segmented = torch.FloatTensor(lung_segmentation_scan(resUnet, scan))
print(scan_segmented.shape)
torch.save(scan_segmented, 'seg.pt')

"""
data_path = "C:/Users/pedro.fernandes.sous/Desktop/Sandbox/hu_generation_samples"
data_filepaths = os.listdir(data_path)
data_filepaths = sorted(data_filepaths, key=lambda x: int(re.search(r'_(\d+)\.pt$', x).group(1)))
for i in range(len(data_filepaths)):
#for i in range(2):
    print(data_filepaths[i])
    scan = load_PT(f"{data_path}/{data_filepaths[i]}")


#--------------SEGMENTATION OF THE WHOLE SCAN--------------
    scan_segmented = torch.FloatTensor(lung_segmentation_scan(resUnet, scan))
    #scan_segmented = scan_segmented.permute(2, 0, 1)
    print(scan_segmented.shape)
    torch.save(scan_segmented, f"seg/sample_{i}.pt")




#--------------SEGMENTATION OF A SINGLE SLICE--------------
# example - segment slice 50
#slice_segmented = lung_segmentation_slice(resUnet, scan[50])


'''
# Uncomment if visualization of the segmentation of the scan is desired.

#--------------VISUALIZE SEGMENTED SCAN----------------
# For each slice of the scan, the slice is submitted to a min-max normalization, using the values -1000 and 400HU as the lower and upper limits.
# Then, the slice is resized to a dimension of 128x128 in the x and y axis. The dimension of the voxel in the z axis is not changed.
# If the final size of the segmented scan is different than 128x128, please make the necessary modifications in the following line regarding the size.
scan_preprocessed = np.array([cv2.resize(normalize(img), (128,128)) for img in scan])

for idx in range(len(scan_segmented)):
    plt.title('Scan & Mask: ' + str(idx))
    plt.imshow(scan_preprocessed[idx], cmap='gray')
    plt.imshow(scan_segmented[idx], cmap='jet', alpha=0.4)
    plt.show()
'''
    
