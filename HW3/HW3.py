# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ELE490 - Fundamentals of Image Processing 
# Assignment 3
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

## Import necessary libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from os import getcwd as gcd



## Definitions and reading the image
# Define path and name to read the image
fileSep = "\\"
imageRootPath = gcd() 
imageRootPath = imageRootPath + fileSep + 'images'

imagePath_Cameraman = str(imageRootPath + fileSep + 'ELE490_Cameraman.tif')
imagePath_noisyCameraman = str(imageRootPath + fileSep + 'ELE490_noisyCameraman.tif')

## Q1 read the images
# Open the image: Cameraman
image_Cameraman = Image.open(imagePath_Cameraman)
# Convert image to grayscale if it's not already in that mode
image_Cameraman = image_Cameraman.convert('L')

# Open the image: noisyCameraman
image_noisyCameraman = Image.open(imagePath_noisyCameraman)
# Convert image to grayscale if it's not already in that mode
image_noisyCameraman = image_noisyCameraman.convert('L')

# Convert images to numpy arrays
arrayU8_Cameraman = np.array(image_Cameraman, dtype=np.uint8)
arrayU8_noisyCameraman = np.array(image_noisyCameraman, dtype=np.uint8)

# Create arrays in float32 type
arrayF32_Cameraman = np.asarray(arrayU8_Cameraman, dtype=np.float32)
arrayF32_noisyCameraman = np.asarray(arrayU8_noisyCameraman, dtype=np.float32)


## Q2 Display both images
# Display the original image: Cameraman
plt.imshow(image_Cameraman, cmap='gray')
plt.axis('off')
plt.title('Original image: Cameraman.')
plt.imsave(str(imageRootPath + fileSep + 'imageCameraman.jpg'), image_Cameraman, cmap='gray')
plt.show()

# Display the original image: noisyCameraman
plt.imshow(image_noisyCameraman, cmap='gray')
plt.axis('off')
plt.title('Original image: noisyCameraman.')
plt.imsave(str(imageRootPath + fileSep + 'imagenoisyCameraman.jpg'), image_noisyCameraman, cmap='gray')
plt.show()


## Q3 Apply some filters
# Create filters: 3x3 unity
filter_H3 = np.ones((3,3), dtype=np.float32) * (1/9)
# Create filters: 4x4 unity
filter_H4 = np.ones((4,4), dtype=np.float32) * (1/16)
# Create filters: 5x5 unity
filter_H5 = np.ones((5,5), dtype=np.float32) * (1/25)

# Filter image: filter the image Cameraman
arrayF32_Cameraman_filteredH3 = signal.convolve2d(arrayF32_Cameraman, 
                                                  filter_H3, 
                                                  mode='full', # perform full 2D discrete convolution 
                                                  boundary='fill', fillvalue=0) # pad by 0  
arrayU8_Cameraman_filteredH3 = np.asarray(arrayF32_Cameraman_filteredH3, dtype=np.uint8)
image_Cameraman_filteredH3 = Image.fromarray(arrayU8_Cameraman_filteredH3).convert('L')

arrayF32_Cameraman_filteredH4 = signal.convolve2d(arrayF32_Cameraman, 
                                                  filter_H4, 
                                                  mode='full', # perform full 2D discrete convolution 
                                                  boundary='fill', fillvalue=0) # pad by 0  
arrayU8_Cameraman_filteredH4 = np.asarray(arrayF32_Cameraman_filteredH4, dtype=np.uint8)
image_Cameraman_filteredH4 = Image.fromarray(arrayU8_Cameraman_filteredH4).convert('L')

arrayF32_Cameraman_filteredH5 = signal.convolve2d(arrayF32_Cameraman, 
                                                  filter_H5, 
                                                  mode='full', # perform full 2D discrete convolution 
                                                  boundary='fill', fillvalue=0) # pad by 0  
arrayU8_Cameraman_filteredH5 = np.asarray(arrayF32_Cameraman_filteredH5, dtype=np.uint8)
image_Cameraman_filteredH5 = Image.fromarray(arrayU8_Cameraman_filteredH5).convert('L')

# Display filtered images: cameraman images
plt.imshow(image_Cameraman_filteredH3, cmap='gray')
plt.axis('off')
plt.title('Filtered image: Cameraman filtered by h_3.')
plt.imsave(str(imageRootPath + fileSep + 'image_Cameraman_filteredH3.jpg'), image_Cameraman, cmap='gray')
plt.show()

plt.imshow(image_Cameraman_filteredH4, cmap='gray')
plt.axis('off')
plt.title('Filtered image: Cameraman filtered by h_4.')
plt.imsave(str(imageRootPath + fileSep + 'image_Cameraman_filteredH4.jpg'), image_Cameraman, cmap='gray')
plt.show()

plt.imshow(image_Cameraman_filteredH5, cmap='gray')
plt.axis('off')
plt.title('Filtered image: Cameraman filtered by h_5.')
plt.imsave(str(imageRootPath + fileSep + 'image_Cameraman_filteredH5.jpg'), image_Cameraman, cmap='gray')
plt.show()

# Filter image: filter the image noisyCameraman
arrayF32_noisyCameraman_filteredH3 = signal.convolve2d(arrayF32_noisyCameraman, 
                                                  filter_H3, 
                                                  mode='full', # perform full 2D discrete convolution 
                                                  boundary='fill', fillvalue=0) # pad by 0  
arrayU8_noisyCameraman_filteredH3 = np.asarray(arrayF32_noisyCameraman_filteredH3, dtype=np.uint8)
image_noisyCameraman_filteredH3 = Image.fromarray(arrayU8_noisyCameraman_filteredH3).convert('L')

arrayF32_noisyCameraman_filteredH4 = signal.convolve2d(arrayF32_noisyCameraman, 
                                                  filter_H4, 
                                                  mode='full', # perform full 2D discrete convolution 
                                                  boundary='fill', fillvalue=0) # pad by 0  
arrayU8_noisyCameraman_filteredH4 = np.asarray(arrayF32_noisyCameraman_filteredH4, dtype=np.uint8)
image_noisyCameraman_filteredH4 = Image.fromarray(arrayU8_noisyCameraman_filteredH4).convert('L')

arrayF32_noisyCameraman_filteredH5 = signal.convolve2d(arrayF32_noisyCameraman, 
                                                  filter_H5, 
                                                  mode='full', # perform full 2D discrete convolution 
                                                  boundary='fill', fillvalue=0) # pad by 0  
arrayU8_noisyCameraman_filteredH5 = np.asarray(arrayF32_noisyCameraman_filteredH5, dtype=np.uint8)
image_noisyCameraman_filteredH5 = Image.fromarray(arrayU8_noisyCameraman_filteredH5).convert('L')

# Display filtered images: noisy cameraman images
plt.imshow(image_noisyCameraman_filteredH3, cmap='gray')
plt.axis('off')
plt.title('Filtered image: noisyCameraman filtered by h_3.')
plt.imsave(str(imageRootPath + fileSep + 'image_noisyCameraman_filteredH3.jpg'), image_noisyCameraman, cmap='gray')
plt.show()

plt.imshow(image_noisyCameraman_filteredH4, cmap='gray')
plt.axis('off')
plt.title('Filtered image: noisyCameraman filtered by h_4.')
plt.imsave(str(imageRootPath + fileSep + 'image_noisyCameraman_filteredH4.jpg'), image_noisyCameraman, cmap='gray')
plt.show()

plt.imshow(image_noisyCameraman_filteredH5, cmap='gray')
plt.axis('off')
plt.title('Filtered image: noisyCameraman filtered by h_5.')
plt.imsave(str(imageRootPath + fileSep + 'image_noisyCameraman_filteredH5.jpg'), image_noisyCameraman, cmap='gray')
plt.show()

