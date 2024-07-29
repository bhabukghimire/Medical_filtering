import SimpleITK as sitk
import numpy as np
from pathlib import Path
import cv2
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import math
from itertools import product
import sys
import os
import nibabel as nib

import warnings
warnings.filterwarnings('ignore')

'''
convolve(input, weights, output=None, mode='reflect', cval=0.0, origin=0)

padding methods: 
reflect 
The input is extended by reflecting about the edge of the last pixel. This mode is also sometimes referred to as half-sample symmetric.

constant
The input is extended by filling all values beyond the edge with the same constant value, defined by the cval parameter.

nearest 
The input is extended by replicating the last pixel.

mirror 
The input is extended by reflecting about the center of the last pixel. This mode is also sometimes referred to as whole-sample symmetric.

wrap
The input is extended by wrapping around to the opposite edge.



'''

class Filtering:
    def __init__(self, image_name, kernel_size):
        self.image_name = image_name
        self.kernel_size = kernel_size
        self.kernel = None
        self.phantom_name = Path(Path(os.path.basename(self.image_name)).stem).stem

        self.image = sitk.GetArrayFromImage(sitk.ReadImage(self.image_name))
    

    def __get_mean_kernel3d(self):
        return np.ones((self.kernel_size,self.kernel_size,self.kernel_size)) / self.kernel_size ** 3
    
    def __get_mean_kernel2d(self):
        return np.ones((self.kernel_size, self.kernel_size)) / self.kernel_size ** 2


    def __compute_weight(self,position,sigma,dim):
        distance_2 = np.sum(position**2)
        # $\frac{-1}{\sigma^2} * \frac{1}{\sqrt{2 \pi} \sigma}^D = \frac{-1}{\sqrt{D/2}{2 \pi} * \sigma^{D+2}}$
        first_part = -1/((2*math.pi)**(dim/2) * sigma ** (dim+2))

        # $(D - \frac{||k||^2}{\sigma^2}) * e^{\frac{-||k||^2}{2 \sigma^2}}$
        second_part = (dim - distance_2/ sigma**2)*math.e**(-distance_2/(2 * sigma**2))

        return first_part * second_part
    
    def __get_LOG_kernel(self,sigma,dim):
        # Initialize the kernel as tensor of zeros
        kernel = np.zeros([self.kernel_size for _ in range(dim)])

        for k in product(range(self.kernel_size), repeat= dim):
            kernel[k] = self.__compute_weight(np.array(k)-int((self.kernel_size-1)/2),sigma,dim)

        kernel -= np.sum(kernel)/np.prod(kernel.shape)
        return kernel


    def mean_filtering3d(self,mode, constant_value,save_output = False,file_name = None):
        self.kernel = self.__get_mean_kernel3d()
        convolved_image = convolve(self.image, self.kernel, mode = mode,cval = constant_value)
        if save_output:
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(convolved_image,affine)
            if file_name:
                nib.save(nifti_img,file_name)
                return convolved_image
            else:
                nib.save(nifti_img,f'{self.phantom_name}_mean3d.nii.gz')
                return convolved_image
        
        else:
            return convolved_image          



    
    def mean_filtering2d(self,mode,constant_value,save_output = False, file_name = None):        
        filtered2d = []
        self.kernel = self.__get_mean_kernel2d()
        for slice_index in range(self.image.shape[0]):
            filtered2d.append(convolve(self.image[slice_index], self.kernel,mode = mode, cval = constant_value))

        convolved_image = np.array(filtered2d)
        if save_output:
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(convolved_image,affine)
            if file_name:
                nib.save(nifti_img,file_name)
                return convolved_image
            else:
                nib.save(nifti_img,f'{self.phantom_name}_mean2d.nii.gz')
                return convolved_image

        else:
            return convolved_image  
        



    def GaborFilter(self,sigma,theta,Lmbda,gamma,psi,mode, constant_value,rotation_invariance = False,save_output = False, file_name = None):

        ksize = 15 # size of the returned filter

        if rotation_invariance:
            num_orientations = round(2*np.pi/abs(theta))
           
            gabor_kernels = []
            for theta in np.linspace(0,np.pi,num_orientations):
                kernl = cv2.getGaborKernel((ksize,ksize),sigma,theta,Lmbda,gamma,psi)
                gabor_kernels.append(kernl)
            
            filtered_images = []
            img_arr = [] # to store the result of convolution of different kernel in single image
            for img_idx in range(self.image.shape[0]):
                for kernel_ in gabor_kernels:
                    img_arr.append(convolve(self.image[img_idx],kernel_, mode = mode,cval = constant_value))
                
                # for average pooling
                filtered_images.append(np.mean(img_arr,axis = 0))
            
            convolved_image = np.array(filtered_images)
            if save_output:
                affine = np.eye(4)
                nifti_img = nib.Nifti1Image(convolved_image,affine)
                if file_name:
                    nib.save(nifti_img,file_name)
                    return convolved_image
                else:
                    nib.save(nifti_img,f'{self.phantom_name}_Gabor.nii.gz')
                    return convolved_image

            else:
                return convolved_image  
            
        
        else:
            filtered_images = []
            kernel_ = cv2.getGaborKernel((ksize,ksize),sigma,theta,Lmbda,gamma,psi)
            for img_idx in range(self.image.shape[0]):
                filtered_images.append(convolve(self.image[img_idx],kernel_,mode = mode, cval = constant_value))
            
            convolved_image = np.array(filtered_images)
            if save_output:
                affine = np.eye(4)
                nifti_img = nib.Nifti1Image(convolved_image,affine)
                if file_name:
                    nib.save(nifti_img,file_name)
                    return convolved_image
                else:
                    nib.save(nifti_img,f'{self.phantom_name}_Gabor.nii.gz')
                    return convolved_image

            else:
                return convolved_image  
            


    def LoG_filtering3d(self,sigma,dim,mode,constant_value ,save_output = False, file_name = None):
        self.kernel = self.__get_LOG_kernel(sigma,dim)
        convolved_image = convolve(self.image,self.kernel,mode = mode, cval = constant_value)

        if save_output:
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(convolved_image,affine)
            if file_name:
                nib.save(nifti_img,file_name)
                return convolved_image
            else:
                nib.save(nifti_img,f'{self.phantom_name}_LoG3d.nii.gz')
                return convolved_image

        else:
            return convolved_image  
        
    

    
    def LoG_filtering2d(self,sigma,dim,mode,constant_value,save_output = False, file_name = None):
        filtered2d = []
        self.kernel = self.__get_LOG_kernel(sigma,dim)

        for slice_index in range(self.image.shape[0]):
            filtered2d.append(convolve(self.image[slice_index],self.kernel,mode = mode, cval = constant_value))
        convolved_image = np.array(filtered2d)
        if save_output:
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(convolved_image,affine)
            if file_name:
                nib.save(nifti_img,file_name)
                return convolved_image
            else:
                nib.save(nifti_img,f'{self.phantom_name}_Log2d.nii.gz')
                return convolved_image

        else:
            return convolved_image  
        





img_path = '/home/bhabuk/Desktop/Filtering/naamiiInternship/test_dataset/sphere/image/sphere.nii.gz'

filter = Filtering(img_path,15)

# for gabor filter
# sigma = 5
# theta = 1 * np.pi/4
# Lambda = 2 / np.pi
# psi = 0
# gamma = 3/2


filtered_image = filter.GaborFilter(20,np.pi/ 8,8,5/2,0,'mirror',0,True,True,"4_b_2_Impulse_Gabor.nii.gz")


i = 10
plt.imshow(filtered_image[i],cmap = 'gray')
plt.axis('off')
plt.show()