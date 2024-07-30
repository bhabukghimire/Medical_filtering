import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

path = '/home/bhabuk/Desktop/Filtering/naamiiInternship/Data_nii/S0819_P200944693_0_CT_5_3_CHEST_ABDOMEN_PELVIS_2.nii.gz'

data = sitk.GetArrayFromImage(sitk.ReadImage(path))

# plt.imshow(data[255,:,:],cmap = 'gray')
# plt.show()
import numpy as np
from vedo import Volume, show


# Create a Volume from the numpy array
volume = Volume(data)

# Display the volume
show(volume, "3d view of image", axes=2)
