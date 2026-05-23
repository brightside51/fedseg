# LIBRARIES
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


# PREPROCESSING FUNCTION: ESSENTIAL TO MAP CT ORIGINAL HU VALUES TO A RANGE BETWEEN 0 AND 1
#----------------MIN-MAX NORMALIZATION----------------
def normalize(scan, minimum = -1000, maximum = 400):
    scan = (scan - minimum) / (maximum - minimum)
    scan[scan >= 1] = 1
    scan[scan <= 0] = 0
    
    return scan




#---------------SEGMENTATION FUNCTIONS-------------
#-------------SEGMENTATION OF THE SCAN----------
def lung_segmentation_scan(model, scan_data, final_img_size = (128, 128), fill_holes = False):
    """
    model:          Segmentation model
    scan_data:      Scan to be segmented
    final_img_size: Desired mask size
    fill_holes:     True if desired filling of the possible holes of the masks
    """

    scan = scan_data
    predicted_masks = []
    for idx in range(scan.shape[0]):
        img = scan[idx]
        
        # Resize image to the input dimension of the network
        img_resized = np.array([cv2.resize(normalize(img), (128,128))])
        
        # Get predicted lung mask
        predicted = model(img_resized)
        
        predicted_mask = np.array(predicted[0])
        p1 = np.array(predicted[0])
        predicted_mask = cv2.resize(p1, final_img_size)
        for r in range(predicted_mask.shape[0]):
            for c in range(predicted_mask.shape[1]):
                if predicted_mask[r][c] >= 0.5:
                    predicted_mask[r][c] = 1
                else:
                    predicted_mask[r][c] = 0
        if fill_holes:
            predicted_mask = binary_fill_holes(predicted_mask)
        predicted_masks.append(predicted_mask)
        #predicted_masks.append(new_mask)
    
    return predicted_masks


#-------------SEGMENTATION OF A SINGLE SLICE----------------
def lung_segmentation_slice(model, slice_data, final_img_size = (128, 128), fill_holes = False):
    img = slice_data
    
    # Resize image to the input dimension of the network
    img_resize = np.array([cv2.resize(normalize(img), (128,128))])
        
    # Get predicted lung mask
    predicted = model(img_resize)
        
    predicted_mask = np.array(predicted[0])
    p1 = np.array(predicted[0])
    predicted_mask = cv2.resize(p1, final_img_size)
    for r in range(predicted_mask.shape[0]):
        for c in range(predicted_mask.shape[1]):
            if predicted_mask[r][c] >= 0.5:
                predicted_mask[r][c] = 1
            else:
                predicted_mask[r][c] = 0
    if fill_holes:
        predicted_mask = binary_fill_holes(predicted_mask)
       
    return predicted_mask
