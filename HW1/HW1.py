# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ELE490 - Fundamentals of Image Processing 
# Assignment 1
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

## Import necessary libraries
import cv2 as cv
# from PIL import Image
import numpy as np

## Definitions and reading the image
# Define path and name to read the image
imageRootPath = "images"
imageName = "mySelfie.jpg"
separator = "\\"

imagePath = str(imageRootPath + separator + imageName)

# Check if image is read and exit if not 
selfie = cv.imread(imagePath, cv.IMREAD_COLOR)
if selfie is None or False: 
    print("Can't read the selfie.")
    print("Exiting...")
    exit()

## Q1 
# Convert the image to a numpy array of 8-bit integers
imgArray = np.array(selfie, dtype=np.uint8)

# Convert the array to float32
imgArray_32 = imgArray.astype(np.float32)

imgArray_red = imgArray[:,:,2]
imgArray_green = imgArray[:,:,1]
imgArray_blue = imgArray[:,:,0]


# Print the dimensions of image (2320x2320)
# print(f"Size: {imgArray.shape[0]}x{imgArray.shape[1]}")

# display the original image (resized for )
cv.imshow("Original Selfie", cv.resize(imgArray, (512,512)))
# cv.waitKey(0)

## Q2 
cv.imshow("Red Channel as Grayscale", cv.resize(imgArray_red, (512,512)))
# cv.waitKey(0)

cv.imshow("Green Channel as Grayscale", cv.resize(imgArray_green, (512,512)))
# cv.waitKey(0)

cv.imshow("Blue Channel as Grayscale", cv.resize(imgArray_blue, (512,512)))
cv.waitKey(0)



