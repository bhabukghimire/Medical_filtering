import math
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.ndimage import convolve,rotate,uniform_filter
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path
import os

import warnings
warnings.filterwarnings('ignore')


class Laws:
    def __init__(self,image_name):
        self.image_name = image_name
        self.laws_kernel_name = None
        self.phantom_name = Path(Path(os.path.basename(self.image_name)).stem).stem

        self.img = sitk.GetArrayFromImage(sitk.ReadImage(self.image_name))

    
    def __get_1D_filter_name(self):
        return np.array([name for name in re.findall(r'[A-Z]\d',self.laws_kernel_name)])

    def __check_pad_requirements(self):
        ls = np.array([int(number) for number in re.findall(r'\d+',self.laws_kernel_name)])
        return not(ls.min() == ls.max())

    def __get_1D_filter(self,name,pad):
        if name == "L3":
            ker = np.array([0, 1, 2, 1, 0]) if pad else np.array([1, 2, 1])
            return 1/math.sqrt(6) * ker
        elif name == "L5":
            return 1/math.sqrt(70) * np.array([1, 4, 6, 4, 1])
        elif name == "E3":
            ker = np.array([0, -1, 0, 1, 0]) if pad else np.array([-1, 0, 1])
            return 1 / math.sqrt(2) * ker
        elif name == "E5":
            return 1 / math.sqrt(10) * np.array([-1, -2, 0, 2, 1])
        elif name == "S3":
            ker = np.array([0, -1, 2, -1, 0]) if pad else np.array([-1, 2, -1])
            return 1 / math.sqrt(6) * ker
        elif name == "S5":
            return 1 / math.sqrt(6) * np.array([-1, 0, 2, 0, -1])
        elif name == "W5":
            return 1 / math.sqrt(10) * np.array([-1, 2, 0, -2, 1])
        elif name == "R5":
            return 1 / math.sqrt(70) * np.array([1, -4, 6, -4, 1])
        else:
            raise Exception(f"{name} is not a valid filter name. "
                            "Choose between : L3, L5, E3, E5, S3, S5, W5 or R5")

    

    def __get_kernel(self,kernel_name):
        self.laws_kernel_name = kernel_name
        pad_req = self.__check_pad_requirements()
        fName1D = self.__get_1D_filter_name() # to get the list of 1D filter name from given kernel name

        if len(fName1D) == 2:
            kernel = np.outer(self.__get_1D_filter(fName1D[1],pad_req), self.__get_1D_filter(fName1D[0],pad_req))
        
        elif len(fName1D) == 3:

            kernel = np.outer(self.__get_1D_filter(fName1D[2],pad_req),
                              np.outer(self.__get_1D_filter(fName1D[1],pad_req), self.__get_1D_filter(fName1D[0],pad_req))).reshape(5,5,5)

        return kernel

   # the kernel are rotated by 45 deg to achieve rotational invariance
    def __rotate_kernels(self,kernel):
        rotated_kernels = []
        for angle in range(0,360,45):
            rotated_kernels.append(rotate(kernel,angle,reshape = False))
        return rotated_kernels


    def __compute_energy_map(self,convolved_img,delta):
        squared_image = convolved_img ** 2
        return uniform_filter(squared_image,size = delta)
    
    def convolve_image(self,kernel_name, mode, constant_value,rotation_invariance = False, pooling = 'max' ,delta = None
                       ,save_output = False, file_name = None):

        kernel = self.__get_kernel(kernel_name)


        if kernel.ndim == 3:
            if rotation_invariance:
                rotated_kernels = self.__rotate_kernels(kernel)
                convolved_images = []
                for rotated_kernel in rotated_kernels:
                    convolved_images.append(convolve(self.img,rotated_kernel,mode = mode, cval = constant_value))

                if pooling == 'max':                    
                    maxpooled = np.max(convolved_images,axis = 0)
                else:
                    maxpooled = convolved_images
                if delta:
                    maxpooled = self.__compute_energy_map(maxpooled,delta)
                    if save_output:
                        affine = np.eye(4)
                        nifti_img = nib.Nifti1Image(maxpooled,affine)
                        if file_name:
                                nib.save(nifti_img,file_name)
                                return maxpooled
                        else:
                                nib.save(nifti_img,f'{self.phantom_name}_{kernel_name}.nii.gz')
                                return maxpooled
                    
                    else:
                        return maxpooled
                
                else:
                    if save_output:
                        affine = np.eye(4)
                        nifti_img = nib.Nifti1Image(maxpooled,affine)
                        if file_name:
                                nib.save(nifti_img,file_name)
                                return maxpooled
                        else:
                                nib.save(nifti_img,f'{self.phantom_name}_{kernel_name}.nii.gz')
                                return maxpooled
                        

                
          
            else:
                convolved_images = convolve(self.img,kernel,mode = mode, cval = constant_value)
                if save_output:
                    affine = np.eye(4)
                    nifti_img = nib.Nifti1Image(convolved_images,affine)
                    if file_name:
                        nib.save(nifti_img,file_name)
                        return convolved_images
                    else:
                        nib.save(nifti_img,f'{self.phantom_name}_{kernel_name}.nii.gz')
                        return convolved_images
                else:
                    return convolved_images

        
        if kernel.ndim == 2:
            if rotation_invariance:
                rotated_kernels = self.__rotate_kernels(kernel)
                filtered_images = []
                
                for slice_index in range(self.img.shape[0]):
                    img_array = []
                    for rotated_kernel in rotated_kernels:
                        img_array.append(convolve(self.img[slice_index],rotated_kernel,mode = mode, cval = constant_value))

                    if pooling == 'max':
                        filtered_images.append(np.max(img_array,axis = 0))
                    else:
                        filtered_images = img_array

                image_ar = np.array(filtered_images)
                if delta:                   
                    convolved_images =  self.__compute_energy_map(image_ar,delta)
                    if save_output:
                        affine = np.eye(4)
                        nifti_img = nib.Nifti1Image(convolved_images,affine)
                        if file_name:
                            nib.save(nifti_img,file_name)
                            return convolved_images
                        else:
                            nib.save(nifti_img,f'{self.phantom_name}_{kernel_name}.nii.gz')
                            return convolved_images
                    else:
                        return convolved_images
                            

                else:
                    if save_output:
                        affine = np.eye(4)
                        nifti_img = nib.Nifti1Image(image_ar,affine)
                        if file_name:
                            nib.save(nifti_img,file_name)
                            return image_ar
                        else:
                            nib.save(nifti_img,f'{self.phantom_name}_{kernel_name}.nii.gz')
                            return image_ar
                    else:
                        return image_ar
                
            else: 

                filtered2d = []
                for slice_index in range(self.img.shape[0]):
                        filtered2d.append(convolve(self.img[slice_index], kernel, mode = mode, cval = constant_value))
                        
                convolved_image = np.array(filtered2d)

                if save_output:
                    affine = np.eye(4)
                    nifti_img = nib.Nifti1Image(convolved_image,affine)
                    if file_name:
                        nib.save(nifti_img,file_name)
                        return convolved_image
                    else:
                        nib.save(nifti_img,f'{self.phantom_name}_{kernel_name}.nii.gz')
                        return convolved_image
                else:
                    return convolved_image
                 



'''
3.c.1 checkerboard • mirror padding
• 2D filter, L5S5 response map
3.c.2 • mirror padding
• 2D filter, L5S5 response map
• 2D rotation invariance, max pooling
3.c.3 • mirror padding
• 2D filter, L5S5 response map
• 2D rotation invariance, max pooling
• energy map, distance  = 7 voxels


'''

img_path = '/home/bhabuk/Desktop/Filtering/naamiiInternship/test_dataset/checkerboard/image/checkerboard.nii.gz'


law = Laws(img_path)
rslt = law.convolve_image('L5S5','mirror',0,False,'none',None,True,"3_c_1_checkerboard_L5S5_mirrorPadding.nii.gz")


print(rslt.shape)
plt.imshow(rslt[50],cmap = 'gray')
plt.show()

