# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ELE490 - Fundamentals of Image Processing 
# Assignment 6
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

## Import necessary libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from os import getcwd as gcd

import numpy as np

"""
Generates zero-mean Gaussian random noise for a given image dimension.
    in:
        dimensions: Tuple indicating the dimensions of the image (N, M).
        sigma: Standard deviation of the Gaussian noise.
    out:
        noise: A 2D numpy array of Gaussian noise.
"""
def generateGaussianNoise(dimensions, sigma):
    N, M = dimensions
    noise = np.random.normal(loc=0, scale=sigma, size=(N, M))  # Zero-mean Gaussian noise
    return noise

## Definitions and reading the image
# Define path and name to read the image
fileSep = "\\"
imageRootPath = gcd() 
imageRootPath = imageRootPath + fileSep + 'images'

imagePath_Cameraman = str(imageRootPath + fileSep + 'ELE490_Cameraman.tif')

## Q1 read the image and scale into range 0-1
# Open the image: Cameraman
img_Cameraman = Image.open(imagePath_Cameraman)
# Convert image to grayscale if it's not already in that mode
img_Cameraman = img_Cameraman.convert('L')
# Store into an array in unsigned 8-bit type
arr_Cameraman = np.array(img_Cameraman, dtype=np.uint8)

# Display original image
plt.imshow(img_Cameraman, cmap='gray')
plt.axis('off')
plt.title('Original image: Cameraman.')
# plt.imsave(str(imageRootPath + fileSep + 'img_Cameraman.jpg'), img_Cameraman, cmap='gray')
plt.show()

# Scale the image into the range [0-1] 
c = (arr_Cameraman - arr_Cameraman.min()) / (arr_Cameraman.max() - arr_Cameraman.min())

## Q2 - Obtain Gaussian-noised images with different sigma values 

# Display the noise for sigma=25 
N,M = arr_Cameraman.shape[0], arr_Cameraman.shape[1]
imgDimensions = (N,M)
sigma_test = 25  # Standard deviation for the noise

arr_gaussianNoise = generateGaussianNoise(imgDimensions, sigma_test)

# Display the generated noise
plt.figure()
plt.title(f"Gaussian Random Noise (sigma={sigma_test})")
plt.imshow(arr_gaussianNoise, cmap='gray')
plt.colorbar(label="Noise Intensity")
plt.imsave(str(imageRootPath + fileSep + 'img_GaussianNoise_sigma_' + str(int(sigma_test)) + '.jpg'), arr_gaussianNoise, cmap='gray')
plt.axis('off')
plt.show()

sigmaVals = np.array([10,100,500,1000]) / 1000
Nc = np.zeros((N,M,sigmaVals.shape[0]))

for i,sigma in enumerate(sigmaVals):
    Nc[:,:,i] = c + generateGaussianNoise(imgDimensions, sigma)
    img_Nc = 255 * Nc # its inefficient to multiply the whole array each time
    # Display the noisy image
    plt.figure()
    plt.title(f"Image With Gaussian Random Noise (sigma={sigma})")
    plt.imshow(Nc[:,:,i], cmap='gray')
    # plt.colorbar(label="Noise Intensity")
    plt.savefig(str(imageRootPath + fileSep + 'fig_noisyCameraman_sigma_' + str(int(sigma*1000)) + 'xE-3.jpg'))
    plt.axis('off')
    plt.show()
