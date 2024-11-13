# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ELE490 - Fundamentals of Image Processing 
# Assignment 5
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

## Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import math
from os import getcwd as gcd
import warnings

'''
Performs scaling to [0-255] in uint8 for image display.
    in:
        originalArray: Input array (numpy.ndarray) that is to be scaled.
    out:
        fullGray: Output array which is the scaled version of the input array in uint8.

*Runtime warnings due to division by zero occures when the input image is uniform. To Prevent any warning due to division by zero, warnings module is used to raise "RuntimeWarning" type warnings as errors which is tolerated in except case by assigning a zeros array to the "fullGray" (return is a full black image).

Example usage: 
    imageFullGray = getFullGrayScale(singleChannelImageArray)
'''
def getFullGrayScale(originalArray:np.ndarray):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        try:
            # extract maximum and minimum 
            maximum = np.max(originalArray)
            minimum = np.min(originalArray) 
            # scale the pixel values
            fullGray = ((originalArray-minimum)/(maximum-minimum))*255
        except Warning: # to catch RuntimeWarning occuring due to divide by zero
            fullGray = np.zeros(originalArray.shape)
    # return the scaled array in type uint8 (for proper display)
    return np.array(fullGray, dtype=np.uint8)


def createLowPassFilter(sigma,size=256):
    g_L = np.zeros((size,size))
    for v in range(size):
        for u in range(size):
            g_L[u,v] = math.exp(-1*((u-128)**2 + (v-128)**2)/(2 * sigma**2))

    return g_L



'''
'''
def createChirpSignal(size=256):
    phi_nm = np.zeros((size,size))
    for m in range(size):
        for n in range(size):
            phi_nm[n,m] = np.pi*((n-128)**2 + (m-128)**2) / size

    t_nm = np.cos(phi_nm.copy())

    return phi_nm, t_nm



## Definitions for reading and saving images
# Define path and name to read the image
fileSep = "\\"
imageRootPath = gcd() 
imageRootPath = imageRootPath + fileSep + "images"

'''
## Q1 - Create g_L[u,v] and g_H[u,v] in fft domain
sigmas = np.array([1, 10, 50, 100])


for sigma in sigmas:
    g_L = createLowPassFilter(sigma)
    g_H = np.ones(np.shape(g_L))-g_L

    img_g_L = getFullGrayScale(g_L)
    # Display
    plt.imshow(img_g_L, cmap="gray")
    plt.axis("off")
    plt.title(f"g_L[u,v], sigma={sigma}")
    plt.imsave(str(imageRootPath + fileSep + "img_gL_sigma_" + str(sigma) + ".jpg"), img_g_L, cmap="gray")
    plt.show()

    img_g_H = getFullGrayScale(g_H)
    # Display
    plt.imshow(img_g_H, cmap="gray")
    plt.axis("off")
    plt.title(f"g_H[u,v], sigma={sigma}")
    plt.imsave(str(imageRootPath + fileSep + "img_gH_sigma_" + str(sigma) + ".jpg"), img_g_H, cmap="gray")
    plt.show()
'''

## Q2 - Create a test signal and display it 
ks = np.array(range(9))
rs = 16*ks

phi_nm, chirp_nm = createChirpSignal(size=256)

img_chirp = getFullGrayScale(chirp_nm)
# Display
plt.imshow(img_chirp, cmap="gray")
plt.axis("off")
plt.title(f"chirp[n,m]")
plt.imsave(str(imageRootPath + fileSep + "img_chirp" + ".jpg"), img_chirp, cmap="gray")
plt.show()

img_phi = getFullGrayScale(phi_nm)
# Display
plt.imshow(img_phi, cmap="gray")
plt.axis("off")
plt.title(f"phi[n,m]")
plt.imsave(str(imageRootPath + fileSep + "img_phi" + ".jpg"), img_phi, cmap="gray")
plt.show()

# phi is created centered at an imaginary (127.5,127.5) index
phi_r = np.array(phi_nm[range(127,256), range(127,256)]) 

phi_deriv_r = np.gradient(phi_r)
print(np.shape(phi_deriv_r))
for r in rs:
    print(f"d/dr at r = {r}: {phi_deriv_r[r]:.3f}")
