# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ELE490 - Fundamentals of Image Processing 
# Assignment 6
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

## Import necessary libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

from os import getcwd as gcd

"""
Performs scaling of an image with arbitrary pixel values to an image having pixel values in the range 0-1. 
Even though the function is intended for images, it is not explicitly implied in any operation, instead, this function scales the values of any collection of data that could be converted into a numpy.array to the 0-1 interval.
    in: 
        img: Input image with arbitrary pixel values. 
    out:
        scaledImage: The image scaled to 0-1 interval. 
"""
def scaleImage(img):
    img = np.array(img, dtype=np.float32)
    scaledImage = (img - img.min()) / (img.max() - img.min())
    return scaledImage

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
    # Zero-mean Gaussian noise
    noise = np.random.normal(loc=0, scale=sigma, size=(N, M)) 
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
# Store into an array in 32-bit float type 
arr_Cameraman = np.array(img_Cameraman, dtype=np.float32)

# Display original image
plt.imshow(img_Cameraman, cmap='gray')
plt.axis('off')
plt.title('Original image: Cameraman.')
plt.imsave(str(imageRootPath + fileSep + 'img_Cameraman.jpg'), img_Cameraman, cmap='gray')
plt.show()

# Scale the image into the range [0-1] 
c = scaleImage(arr_Cameraman)

## Q2 - Obtain Gaussian-noised images with different sigma values 
N,M = arr_Cameraman.shape[0], arr_Cameraman.shape[1]
imgDimensions = (N,M)
sigma_test = 10  
# Create the 2D noise of the specified sigma value
arr_gaussianNoise = generateGaussianNoise(imgDimensions, sigma_test)
# Display the generated noise
plt.figure()
plt.title(f"Gaussian Random Noise (sigma={sigma_test})")
plt.axis('off')
plt.imshow(arr_gaussianNoise, cmap='gray')
plt.colorbar(label="Noise Intensity")
plt.savefig(str(imageRootPath + fileSep + 'fig_GaussianNoise_sigma_' + str(int(sigma_test)) + '.jpg'))
plt.show()

# Create an array for different sigma values
sigmaVals = np.array([10,100,500,1000]) / 1000
Nc = np.zeros((N,M,sigmaVals.shape[0]))

for i,sigma in enumerate(sigmaVals):
    Nc[:,:,i] = c + generateGaussianNoise(imgDimensions, sigma)
    img_Nc = 255 * scaleImage(Nc[:,:,i]) 
    # Display the noisy image
    plt.figure()
    plt.title(f"Image With Gaussian Random Noise (sigma={sigma})")
    plt.axis("off")
    plt.imshow(img_Nc, cmap='gray')
    # plt.colorbar(label="Noise Intensity")
    plt.savefig(str(imageRootPath + fileSep + 'fig_noisyCameraman_sigma_' + str(int(sigma*1000)) + 'xE-3.jpg'))
    plt.axis('off')
    plt.show()

## Q3 - Apply adaptive wiener filtering and observe the effect of different
## kernel sizes and sigma estimates.
c_hat = np.zeros((N,M,sigmaVals.shape[0]))
kernel_size = 32
# Loop through 
for i,sigma_noise in enumerate(sigmaVals):
    for sigma_filter in sigmaVals:
        c_hat[:,:,i] = scaleImage(wiener(Nc[:,:,i], mysize=(kernel_size,kernel_size), noise=sigma_filter**2))*255
        # Display the noisy image
        plt.figure()
        plt.title(f"Denoised Image (Noise power:sigma={sigma_noise} Estimate:sigma={sigma_filter})")
        plt.axis("off")
        plt.imshow(c_hat[:,:,i], cmap='gray')
        # plt.colorbar(label="Noise Intensity")
        plt.savefig(str(imageRootPath + fileSep + f"fig_estimateCameraman_noise-sigma_{int(sigma_noise*1000)}xE-3_estimate-sigma_{int(sigma_filter*1000)}xE-3_kernel_{kernel_size}x{kernel_size}" + '.jpg'))
        plt.axis('off')
        plt.show()


