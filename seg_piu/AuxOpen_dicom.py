import os
import pydicom
import numpy as np
import torch

def load_DICOM(data_path):
    # List CT slices files
    ct_dcms = os.listdir(data_path)

    # List the DICOM slice files that are read with pydicom.read_file()
    slices = [pydicom.dcmread(data_path + dcm, force=True) for dcm in ct_dcms]
    
    # Order list of slices in an ascendant way by the position z of the slice
    image = torch.empty((len(ct_dcms), 128, 128)); n_list = []
    for i in range(len(ct_dcms)):
        try:
            n = slices[i].InstanceNumber; n_list.append(n)
            image[n] = torch.nn.functional.interpolate(
                torch.Tensor(slices[i].pixel_array.astype(np.int16)).unsqueeze(0).unsqueeze(0),
                size=(128, 128), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            #print(f"{ct_dcms[i]} -> N : {n}")
        except AttributeError: pass
    n_list = sorted(list(set(n_list)))
    image = image[n_list]
    torch.save(image, "ct.pt")
    #image = image.permute(1, 2, 0)
    
    # Order list of slices in an ascendant way by the position z of the slice
    #slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    #image = np.stack([s.pixel_array for s in slices])
    #image = image.astype(np.int16)
    image[image == -2000] = 0
        
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
                
    image += np.int16(intercept)
    image = np.array(image, dtype=np.int16)

    return image

def load_PT(data_path):

    slices = torch.load(data_path).numpy()
    print(slices.shape)
    print(np.max(slices))
    print(np.min(slices))
    image = torch.empty((slices.shape[0], 128, 128)); n_list = []
    for i in range(slices.shape[0]):
        #v_p = (v - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        s = (slices[i] - 0)/(1 - 0)*(4096 - (-2000)) + (-2000)
        image[i] = torch.nn.functional.interpolate(
            torch.Tensor(s.astype(np.int16)).unsqueeze(0).unsqueeze(0),
            size=(128, 128), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            #print(f"{ct_dcms[i]} -> N : {n}")
    
    torch.save(image, "ct.pt")
    print(image.shape)
    print(torch.max(image))
    print(torch.min(image))
    #image = image.permute(1, 2, 0)
    
    # Order list of slices in an ascendant way by the position z of the slice
    #slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    #image = np.stack([s.pixel_array for s in slices])
    #image = image.astype(np.int16)
    image[image == -2000] = 0
        
    #intercept = slices[0].RescaleIntercept
    #slope = slices[0].RescaleSlope

    #if slope != 1:
    #    image = slope * image.astype(np.float64)
    #    image = image.astype(np.int16)
                
    #image += np.int16(intercept)
    image = np.array(image, dtype=np.int16)

    return image