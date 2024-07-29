import SimpleITK as sitk
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def constant_value_pad(image,  pad_width = 1, constant_value = 1):
    return np.pad(image,pad_width, mode = 'constant', constant_values = constant_value)

def zero_pad(image, pad_width = 1):
    return np.pad(image, pad_width = pad_width, mode = 'constant', constant_values = 0)

def mirror_pad(image, pad_width = 1):
    return np.pad(image, pad_width = pad_width, mode = 'reflect')

def periodic_pad(image, pad_width = 1):
    return np.pad(image, pad_width = pad_width, mode = 'wrap')