# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ELE490 - Fundamentals of Image Processing 
# Assignment 1
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

## Import necessary libraries
import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


## Definitions and reading the image
# Define path and name to read the image
imageRootPath = "images"
# imageName = "myselfie.jpg"
imageName = "rgbPalette.png"
fileSep = "\\"

imagePath = str(imageRootPath + fileSep + imageName)

# Open the image
imageIn = Image.open(imagePath)
# Convert image to RGB if it's not already in that mode
imageIn = imageIn.convert('RGB')

## Q1 
# Convert the image to a numpy array of 8-bit integers
imgArray = np.array(imageIn, dtype=np.uint8)

# Obtain single-channel arrays
imgArray_red = imgArray[:,:,0]
imgArray_green = imgArray[:,:,1]
imgArray_blue = imgArray[:,:,2]

# Create grayscale images from each color channel
img_red = Image.fromarray(imgArray_red, mode='L') # create grayscale image
# convert grayscale image to type 'F' (float32)
imgArray32_red = imageIn.convert('F') 
# assign grayscale image to a numpy array 
imgArray32_red = np.array(imgArray32_red, dtype=np.float32) 

img_green = Image.fromarray(imgArray_green, mode='L') 
imgArray32_green = imageIn.convert('F') 
imgArray32_green = np.array(imgArray32_green, dtype=np.float32) 

img_blue = Image.fromarray(imgArray_blue, mode='L') 
imgArray32_blue = imageIn.convert('F') 
imgArray32_blue = np.array(imgArray32_blue, dtype=np.float32) 

# Get the dimensions of image (2320x2320)
imageHeight = imgArray.shape[0]
imageWidth  = imgArray.shape[1]
print(f"Image dimensions: {imageHeight}x{imageWidth}")


## Q2 
# Display the original image 
plt.imshow(imageIn)
plt.axis('off')
plt.title('Original image.')
plt.show()

# Display each color channel as grayscale images
plt.imshow(img_red, cmap='gray')
plt.axis('off')
plt.title('Red channel, displayed as grayscale.')
plt.imsave(str(imageRootPath + fileSep + 'redCh_gray.png'), img_red)
plt.show()
plt.imshow(img_green, cmap='gray')
plt.axis('off')
plt.title('Green channel, displayed as grayscale.')
plt.imsave(str(imageRootPath + fileSep + 'GreenCh_gray.png'), img_green)
plt.show()
plt.imshow(img_blue, cmap='gray')
plt.axis('off')
plt.title('Blue channel, displayed as grayscale.')
plt.imsave(str(imageRootPath + fileSep + 'BlueCh_gray.png'), img_blue)
plt.show()


## Q3
# Define min threshold values for each channel
redMinThVal = 128
greenMinThVal = 128
blueMinThVal = 128
# Create empty lists to store the index whose values are below the threshold
redMinIndexesToChange = []
greenMinIndexesToChange = []
blueMinIndexesToChange = []
# Create arrays which thresholding will be applied
minThImgArr_red = imgArray_red
minThImgArr_green = imgArray_green
minThImgArr_blue = imgArray_blue

# Applying min-thresholding on R-channel
# Extract the indexes whose values are below the threshold 
redMinIndexesToChange = np.where(minThImgArr_red <= redMinThVal)
if len(redMinIndexesToChange) != 0: # if list of indexes is not empty
    minThImgArr_red[redMinIndexesToChange] = np.uint8(0) 

minThImg_red = Image.fromarray(minThImgArr_red, mode='L') 

plt.imshow(minThImg_red, cmap='gray')
plt.axis('off')
plt.title('Min-thresholded red channel with th='+str(redMinThVal)+', displayed as grayscale')
plt.imsave(str(imageRootPath + fileSep + 'minThRedCh_gray.png'), minThImg_red, cmap='gray')
plt.show()

# Applying min-thresholding on G-channel
greenMinIndexesToChange = np.where(minThImgArr_green <= greenMinThVal)
if len(greenMinIndexesToChange) != 0: 
    minThImgArr_green[greenMinIndexesToChange] = np.uint8(0) 

minThImg_green = Image.fromarray(minThImgArr_green, mode='L') 

plt.imshow(minThImg_green, cmap='gray')
plt.axis('off')
plt.title('Min-thresholded green channel with th='+str(greenMinThVal)+', displayed as grayscale')
plt.imsave(str(imageRootPath + fileSep + 'minThGreenCh_gray.png'), minThImg_green)
plt.show()

# Applying min-thresholding on B-channel
blueMinIndexesToChange = np.where(minThImgArr_blue <= blueMinThVal)
if len(blueMinIndexesToChange) != 0: 
    minThImgArr_blue[blueMinIndexesToChange] = np.uint8(0) 

minThImg_blue = Image.fromarray(minThImgArr_blue, mode='L') 

plt.imshow(minThImg_blue, cmap='gray')
plt.axis('off')
plt.title('Min-thresholded blue channel with th='+str(blueMinThVal)+', displayed as grayscale')
plt.imsave(str(imageRootPath + fileSep + 'minThBlueCh_gray.png'), minThImg_blue)
plt.show()


## Q4
# Define max threshold values for each channel
redMaxThVal = 128
greenMaxThVal = 128
blueMaxThVal = 128
# Create empty lists to store the index whose values are below the threshold
redMaxIndexesToChange = []
greenMaxIndexesToChange = []
blueMaxIndexesToChange = []
# Create arrays which thresholding will be applied
maxThImgArr_red = imgArray_red
maxThImgArr_green = imgArray_green
maxThImgArr_blue = imgArray_blue

# Applying max-thresholding on R-channel
# Extract the indexes whose values are below the threshold 
redMaxIndexesToChange = np.where(maxThImgArr_red >= redMaxThVal)
if len(redMaxIndexesToChange) != 0: # if list of indexes is not empty
    maxThImgArr_red[redMaxIndexesToChange] = np.uint8(255) 

maxThImg_red = Image.fromarray(maxThImgArr_red, mode='L') 

plt.imshow(maxThImg_red, cmap='gray')
plt.axis('off')
plt.title('Max-thresholded red channel with th='+str(redMaxThVal)+', displayed as grayscale')
plt.imsave(str(imageRootPath + fileSep + 'maxThRedCh_gray.png'), maxThImg_red, cmap='gray')
plt.show()

# Applying max-thresholding on G-channel
greenMaxIndexesToChange = np.where(maxThImgArr_green >= greenMaxThVal)
if len(greenMaxIndexesToChange) != 0: 
    maxThImgArr_green[greenMaxIndexesToChange] = np.uint8(255) 

maxThImg_green = Image.fromarray(maxThImgArr_green, mode='L') 

plt.imshow(maxThImg_green, cmap='gray')
plt.axis('off')
plt.title('Max-thresholded green channel with th='+str(greenMaxThVal)+', displayed as grayscale')
plt.imsave(str(imageRootPath + fileSep + 'maxThGreenCh_gray.png'), maxThImg_green)
plt.show()

# Applying max-thresholding on B-channel
blueMaxIndexesToChange = np.where(maxThImgArr_blue >= blueMaxThVal)
if len(blueMaxIndexesToChange) != 0: 
    maxThImgArr_blue[blueMaxIndexesToChange] = np.uint8(255) 

maxThImg_blue = Image.fromarray(maxThImgArr_blue, mode='L') 

plt.imshow(maxThImg_blue, cmap='gray')
plt.axis('off')
plt.title('Max-thresholded blue channel with th='+str(blueMaxThVal)+', displayed as grayscale')
plt.imsave(str(imageRootPath + fileSep + 'maxThBlueCh_gray.png'), maxThImg_blue)
plt.show()

