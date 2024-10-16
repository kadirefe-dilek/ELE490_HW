# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ELE490 - Fundamentals of Image Processing 
# Assignment 2
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
## Import necessary libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import getcwd as gcd


## Definitions and reading the image
# Define path and name to read the image
fileSep = "\\"
imageRootPath = gcd() 
imageRootPath = imageRootPath + fileSep + 'images'

imagePathLisa = str(imageRootPath + fileSep + 'ELE490_lisa.tif')
imagePathCameraman = str(imageRootPath + fileSep + 'ELE490_Cameraman.tif')

# Open the image: Lisa
imageLisa = Image.open(imagePathLisa)
# Convert image to grayscale if it's not already in that mode
imageLisa = imageLisa.convert('L')

# Open the image: Cameraman
imageCameraman = Image.open(imagePathCameraman)
# Convert image to grayscale if it's not already in that mode
imageCameraman = imageCameraman.convert('L')

# Convert images to numpy arrays
imgLisaArray = np.array(imageLisa, dtype=np.uint8)
imgCameramanArray = np.array(imageCameraman, dtype=np.uint8)


## Q2 Display both images
# Display the original image: Lisa
plt.imshow(imageLisa, cmap='gray')
plt.axis('off')
plt.title('Original image: Lisa.')
plt.show()

# Display the original image: Cameraman
plt.imshow(imageCameraman, cmap='gray')
plt.axis('off')
plt.title('Original image: Cameraman.')
plt.show()


## Q3 Create Histograms
# Create and plot the histogram of the image, Lisa
histLisa = np.zeros(256)
for i in range(0,256):
    histLisa[i] = np.count_nonzero(imgLisaArray == i)


plt.plot(range(0,256), histLisa)
plt.title('Histogram of the image: Lisa')
plt.xlim(0, 255)
plt.show()

# Create and plot the histogram of the image, Cameraman
histCameraman = np.zeros(256)
for i in range(0,256):
    histCameraman[i] = np.count_nonzero(imgCameramanArray == i)


plt.plot(range(0,256), histCameraman)
plt.title('Histogram of the image: Cameraman')
plt.xlim(0, 255)
plt.show()

## Q4 Match histogram of Lisa to Cameraman


## Q5 Match histogram of Cameraman to Lisa


## Q6 Equalize the histogram of Lisa


## Q7 Equalize the histogram of Lisa

