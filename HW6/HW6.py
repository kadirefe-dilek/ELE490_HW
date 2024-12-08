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
