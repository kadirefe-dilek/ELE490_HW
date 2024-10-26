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


'''
Performs filtering by using convolution method present in scipy.signal
    in: 
        arrF32_imageToFilter: The input image that the filtering operation is to be performed on. Expected type is a 2D array and the argument is used as np.array of type float32.
    
        filter: The filter that is to be used for the filtering operation. Expected type is a 2D array and the argument is used as np.array of type float32.
    out:
        img_imageFiltered: Image object that includes the filtered image.

        arrF32_imageFiltered: Numpy array of type float32 that carries the pixel values of the filtered image. 
'''
def filterImage(arrF32_imageToFilter:np.array, filter):
    # Convert input arrays to np arrays of type float32
    arrF32_imageToFilter = np.asarray(arrF32_imageToFilter, 
                                        dtype=np.float32)
    filter = np.asarray(filter, dtype=np.float32)
    # Perform convolution 
    arrF32_imageFiltered = signal.convolve2d(
                            arrF32_imageToFilter, filter, # arg1 * arg2
                            mode='full', # perform full 2D discrete convolution
                            boundary='fill', fillvalue=0 # pad by 0 
                            ) 
    # Create an array of type uint8 to create the image from
    arrU8_imageFiltered = np.asarray(arrF32_imageFiltered, dtype=np.uint8)
    # Create an Image object from filtered image
    img_imageFiltered = Image.fromarray(arrU8_imageFiltered).convert('L')
    # Return new image both as Image object and float32 np array
    return img_imageFiltered, arrF32_imageFiltered


'''
Performs median filtering using medfilt2d method present in scipy.signal
    in:
        imageToFilter: Numpy array containing the pixel values of the image that the filtering operation is to be performed on. 2D array is expected. 

        filterSize: Integer number which defines the size of the square shaped kernel that is used in median filtering. Default value is 3. 
    out:
        img_imageFiltered: Image object that includes the filtered image. 
'''
def applyMedianFilter(arrU8_imageToFilter:np.array, filterSize:int=3):
    arrU8_filteredImage = np.asarray(signal.medfilt2d(arrU8_imageToFilter, 
                                    kernel_size=filterSize), dtype=np.uint8)

    img_imageFiltered = Image.fromarray(arrU8_filteredImage, mode='L')
    return img_imageFiltered


## Definitions and reading the image
# Define path and name to read the image
fileSep = "\\"
imageRootPath = gcd() 
imageRootPath = imageRootPath + fileSep + 'images'

imagePath_Cameraman = str(imageRootPath + fileSep + 'ELE490_Cameraman.tif')
imagePath_noisyCameraman = str(imageRootPath + fileSep + 'ELE490_noisyCameraman.tif')

## Q1 read the images
# Open the image: Cameraman
img_Cameraman = Image.open(imagePath_Cameraman)
# Convert image to grayscale if it's not already in that mode
img_Cameraman = img_Cameraman.convert('L')

# Open the image: noisyCameraman
img_noisyCameraman = Image.open(imagePath_noisyCameraman)
# Convert image to grayscale if it's not already in that mode
img_noisyCameraman = img_noisyCameraman.convert('L')

# Convert images to numpy arrays
arrU8_Cameraman = np.array(img_Cameraman, dtype=np.uint8)
arrU8_noisyCameraman = np.array(img_noisyCameraman, dtype=np.uint8)

# Create arrays in float32 type
arrF32_Cameraman = np.asarray(arrU8_Cameraman, dtype=np.float32)
arrF32_noisyCameraman = np.asarray(arrU8_noisyCameraman, dtype=np.float32)


## Q2 Display both images
# Display the original image: Cameraman
plt.imshow(img_Cameraman, cmap='gray')
plt.axis('off')
plt.title('Original image: Cameraman.')
plt.imsave(str(imageRootPath + fileSep + 'img_Cameraman.jpg'), img_Cameraman, cmap='gray')
plt.show()

# Display the original image: noisyCameraman
plt.imshow(img_noisyCameraman, cmap='gray')
plt.axis('off')
plt.title('Original image: noisyCameraman.')
plt.imsave(str(imageRootPath + fileSep + 'img_noisyCameraman.jpg'), img_noisyCameraman, cmap='gray')
plt.show()


## Q3 Apply some filters
# Create filters: 3x3 unity
filter_H3 = np.ones((3,3), dtype=np.float32) * (1/9)
# Create filters: 4x4 unity
filter_H4 = np.ones((4,4), dtype=np.float32) * (1/16)
# Create filters: 5x5 unity
filter_H5 = np.ones((5,5), dtype=np.float32) * (1/25)

# Filter image: filter the image Cameraman
img_Cameraman_filteredH3, arrF32_Cameraman_filteredH3 = filterImage(
                                                            arrF32_Cameraman, filter_H3)

img_Cameraman_filteredH4, arrF32_Cameraman_filteredH4 = filterImage(
                                                            arrF32_Cameraman, filter_H4)

img_Cameraman_filteredH5, arrF32_Cameraman_filteredH5 = filterImage(
                                                            arrF32_Cameraman, filter_H5)

# Display filtered images: cameraman images
plt.imshow(img_Cameraman_filteredH3, cmap='gray')
plt.imshow(img_Cameraman_filteredH3, cmap='gray')
plt.axis('off')
plt.title('Filtered image: Cameraman filtered by h_3.')
plt.imsave(str(imageRootPath + fileSep + 'img_Cameraman_filteredH3.jpg'), img_Cameraman_filteredH3, cmap='gray')
plt.show()

plt.imshow(img_Cameraman_filteredH4, cmap='gray')
plt.imshow(img_Cameraman_filteredH4, cmap='gray')
plt.axis('off')
plt.title('Filtered image: Cameraman filtered by h_4.')
plt.imsave(str(imageRootPath + fileSep + 'img_Cameraman_filteredH4.jpg'), img_Cameraman_filteredH4, cmap='gray')
plt.show()

plt.imshow(img_Cameraman_filteredH5, cmap='gray')
plt.imshow(img_Cameraman_filteredH5, cmap='gray')
plt.axis('off')
plt.title('Filtered image: Cameraman filtered by h_5.')
plt.imsave(str(imageRootPath + fileSep + 'img_Cameraman_filteredH5.jpg'), img_Cameraman_filteredH5, cmap='gray')
plt.show()

# Filter image: filter the image noisyCameraman
img_noisyCameraman_filteredH3, arrF32_noisyCameraman_filteredH3 =  filterImage(arrF32_noisyCameraman, filter_H3)

img_noisyCameraman_filteredH4, arrF32_noisyCameraman_filteredH4 =  filterImage(arrF32_noisyCameraman, filter_H4)

img_noisyCameraman_filteredH5, arrF32_noisyCameraman_filteredH5 =  filterImage(arrF32_noisyCameraman, filter_H5)

# Display filtered images: noisy cameraman images
plt.imshow(img_noisyCameraman_filteredH3, cmap='gray')
plt.axis('off')
plt.title('Filtered image: noisyCameraman filtered by h_3.')
plt.imsave(str(imageRootPath + fileSep + 'img_noisyCameraman_filteredH3.jpg'), img_noisyCameraman_filteredH3, cmap='gray')
plt.show()

plt.imshow(img_noisyCameraman_filteredH4, cmap='gray')
plt.imshow(img_noisyCameraman_filteredH4, cmap='gray')
plt.axis('off')
plt.title('Filtered image: noisyCameraman filtered by h_4.')
plt.imsave(str(imageRootPath + fileSep + 'img_noisyCameraman_filteredH4.jpg'), img_noisyCameraman_filteredH4, cmap='gray')
plt.show()

plt.imshow(img_noisyCameraman_filteredH5, cmap='gray')
plt.axis('off')
plt.title('Filtered image: noisyCameraman filtered by h_5.')
plt.imsave(str(imageRootPath + fileSep + 'img_noisyCameraman_filteredH5.jpg'), img_noisyCameraman_filteredH5, cmap='gray')
plt.show()


## Q4 - Apply Median filtering
# Filter image: apply median filtering to image Cameraman
img_Cameraman_filteredMed3 = applyMedianFilter(arrU8_Cameraman, 3)
img_Cameraman_filteredMed5 = applyMedianFilter(arrU8_Cameraman, 5)

# Display filtered images: Cameraman images
plt.imshow(img_Cameraman_filteredMed3, cmap='gray')
plt.axis('off')
plt.title('Filtered image: Cameraman filtered by 3x3 median filter.')
plt.imsave(str(imageRootPath + fileSep + 'img_Cameraman_filteredMed3.jpg'), img_Cameraman_filteredMed3, cmap='gray')
plt.show()

plt.imshow(img_Cameraman_filteredMed5, cmap='gray')
plt.axis('off')
plt.title('Filtered image: Cameraman filtered by 5x5 median filter.')
plt.imsave(str(imageRootPath + fileSep + 'img_Cameraman_filteredMed5.jpg'), img_Cameraman_filteredMed5, cmap='gray')
plt.show()

# Filter image: apply median filtering to image noisyCameraman
img_noisyCameraman_filteredMed3 = applyMedianFilter(arrU8_noisyCameraman, 3)
img_noisyCameraman_filteredMed5 = applyMedianFilter(arrU8_noisyCameraman, 5)

# Display filtered images: noisyCameraman images
plt.imshow(img_noisyCameraman_filteredMed3, cmap='gray')
plt.axis('off')
plt.title('Filtered image: noisyCameraman filtered by 3x3 median filter.')
plt.imsave(str(imageRootPath + fileSep + 'img_noisyCameraman_filteredMed3.jpg'), img_noisyCameraman_filteredMed3, cmap='gray')
plt.show()

plt.imshow(img_noisyCameraman_filteredMed5, cmap='gray')
plt.axis('off')
plt.title('Filtered image: noisyCameraman filtered by 5x5 median filter.')
plt.imsave(str(imageRootPath + fileSep + 'img_noisyCameraman_filteredMed5.jpg'), img_noisyCameraman_filteredMed5, cmap='gray')
plt.show()


## Q5 - Apply some filters
# Laplacian filter
filter_L4 = np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]], dtype=np.float32) 

filter_L8 = np.array([[0,  1, 0],
                      [1, -8, 1],
                      [0,  1, 0]], dtype=np.float32)
# Sabel operator (for vertical edge detection)
filter_Sv = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]], dtype=np.float32) 
# Sabel operator (for horizontal edge detection)
filter_Sh = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float32)

# Filter image: filter the image Cameraman by L4, L8, Sv, and Sh 
img_Cameraman_filteredL4, arrF32_Cameraman_filteredL4 = filterImage(
    arrF32_Cameraman, filter_L4)

img_Cameraman_filteredL8, arrF32_Cameraman_filteredL8 = filterImage(
    arrF32_Cameraman, filter_L8)

img_Cameraman_filteredSv, arrF32_Cameraman_filteredSv = filterImage(
    arrF32_Cameraman, filter_Sv)

img_Cameraman_filteredSh, arrF32_Cameraman_filteredSh = filterImage(
    arrF32_Cameraman, filter_Sh)

# Display filtered images: filtered Cameraman images
plt.imshow(img_Cameraman_filteredL4, cmap='gray')
plt.axis('off')
plt.title('Filtered image: Cameraman filtered by L_4.')
plt.imsave(str(imageRootPath + fileSep + 'img_Cameraman_filteredL4.jpg'), img_Cameraman_filteredL4, cmap='gray')
plt.show()

plt.imshow(img_Cameraman_filteredL8, cmap='gray')
plt.axis('off')
plt.title('Filtered image: Cameraman filtered by L_8.')
plt.imsave(str(imageRootPath + fileSep + 'img_Cameraman_filteredL8.jpg'), img_Cameraman_filteredL8, cmap='gray')
plt.show()

plt.imshow(img_Cameraman_filteredSv, cmap='gray')
plt.axis('off')
plt.title('Filtered image: Cameraman filtered by S_v.')
plt.imsave(str(imageRootPath + fileSep + 'img_Cameraman_filteredSv.jpg'), img_Cameraman_filteredSv, cmap='gray')
plt.show()

plt.imshow(img_Cameraman_filteredSh, cmap='gray')
plt.axis('off')
plt.title('Filtered image: Cameraman filtered by S_h.')
plt.imsave(str(imageRootPath + fileSep + 'img_Cameraman_filteredSh.jpg'), img_Cameraman_filteredSh, cmap='gray')
plt.show()

# Filter image: filter the image noisyCameraman by L4, L8, Sv, and Sh 
img_noisyCameraman_filteredL4, arrF32_noisyCameraman_filteredL4 = filterImage(arrF32_noisyCameraman, filter_L4)

img_noisyCameraman_filteredL8, arrF32_noisyCameraman_filteredL8 = filterImage(arrF32_noisyCameraman, filter_L8)

img_noisyCameraman_filteredSv, arrF32_noisyCameraman_filteredSv = filterImage(arrF32_noisyCameraman, filter_Sv)

img_noisyCameraman_filteredSh, arrF32_noisyCameraman_filteredSh = filterImage(arrF32_noisyCameraman, filter_Sh)

# Display filtered images: filtered noisyCameraman images
plt.imshow(img_noisyCameraman_filteredL4, cmap='gray')
plt.axis('off')
plt.title('Filtered image: noisyCameraman filtered by L_4.')
plt.imsave(str(imageRootPath + fileSep + 'img_noisyCameraman_filteredL4.jpg'), img_noisyCameraman_filteredL4, cmap='gray')
plt.show()

plt.imshow(img_noisyCameraman_filteredL8, cmap='gray')
plt.axis('off')
plt.title('Filtered image: noisyCameraman filtered by L_8.')
plt.imsave(str(imageRootPath + fileSep + 'img_noisyCameraman_filteredL8.jpg'), img_noisyCameraman_filteredL8, cmap='gray')
plt.show()

plt.imshow(img_noisyCameraman_filteredSv, cmap='gray')
plt.axis('off')
plt.title('Filtered image: noisyCameraman filtered by S_v.')
plt.imsave(str(imageRootPath + fileSep + 'img_noisyCameraman_filteredSv.jpg'), img_noisyCameraman_filteredSv, cmap='gray')
plt.show()

plt.imshow(img_noisyCameraman_filteredSh, cmap='gray')
plt.axis('off')
plt.title('Filtered image: noisyCameraman filtered by S_h.')
plt.imsave(str(imageRootPath + fileSep + 'img_noisyCameraman_filteredSh.jpg'), img_noisyCameraman_filteredSh, cmap='gray')
plt.show()
