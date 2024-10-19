# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ELE490 - Fundamentals of Image Processing 
# Assignment 2
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

## Import necessary libraries
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import getcwd as gcd

# Functions
'''
Finds the pdf (histogram) of a gray scale image assuming the pixels are having values between 0-255 (image is in type uint8).
    in:
        src_image: source image (2D numpy array)
    out:
        imagePdf: pdf of the image (1D numpy array of length 256)
'''
def getImagePDF(src_image:np.ndarray):
    imagePdf = np.zeros((256))
    totalSize = src_image.flatten().size
    for i in range(256):
        imagePdf[i] = np.count_nonzero(src_image == i) / totalSize

    return np.array(imagePdf)


'''
Matches the histogram of the source image to the reference image's PDF (probability density function).
    in:
        src_image: The input image (numpy array) that is to be modified through histogram matching
        
        reference_pdf: The PDF of the reference image (1D numpy array).
    out: 
        matched_image: Output image array with the histogram matched to the reference PDF. (numpy array with the same shape and type as the src_image)
'''
def matchHistogram(src_image, ref_hist):
    # Calculate the PDF of the source image
    src_hist = getImagePDF(src_image)

    # Calculate the CDF of the source and reference images (CDF is normalized since getImagePDF() returns normalized PDF)
    src_cdf = np.cumsum(src_hist)
    ref_cdf = np.cumsum(ref_hist)

    # Create a lookup table to map source pixels to reference pixels
    lookup_table = np.zeros(256)

    # Map source image pixel values based on CDFs
    for src_value in range(256):
        # Find the closest reference value in the reference CDF for the current source value
        closest_ref_value = np.argmin(np.abs(ref_cdf - src_cdf[src_value]))
        lookup_table[src_value] = closest_ref_value

    # Apply the lookup table to the source image (create matched image from lookup_table indexed by corresponding pixel values)
    matched_image = np.array(lookup_table[src_image.flatten().astype(np.uint8)])

    # Reshape the output image to the original source image shape
    matched_image = matched_image.reshape(src_image.shape)

    return matched_image.astype(src_image.dtype)



## Definitions and reading the image
# Define path and name to read the image
fileSep = "\\"
imageRootPath = gcd() 
imageRootPath = imageRootPath + fileSep + 'images'

imagePathLisa = str(imageRootPath + fileSep + 'ELE490_lisa.tif')
imagePathCameraman = str(imageRootPath + fileSep + 'ELE490_Cameraman.tif')

## Q1 read the images
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
plt.imsave(str(imageRootPath + fileSep + 'imageLisa.jpg'), imageLisa, cmap='gray')
plt.show()

# Display the original image: Cameraman
plt.imshow(imageCameraman, cmap='gray')
plt.axis('off')
plt.title('Original image: Cameraman.')
plt.imsave(str(imageRootPath + fileSep + 'imageCameraman.jpg'), imageCameraman, cmap='gray')
plt.show()


## Q3 Create Histograms and plot them
# Create and plot the histogram of the image, Lisa
histLisa = getImagePDF(imgLisaArray)

_, stemLines, _ = plt.stem(range(0,256), histLisa)
plt.title('Histogram of the image: Lisa')
plt.xlim(0, 255)
plt.setp(stemLines, 'linewidth', 0.5)
plt.savefig(str(imageRootPath + fileSep + 'histLisa_orig.jpg'))
plt.show()

# Create and plot the histogram of the image, Cameraman
histCameraman = getImagePDF(imgCameramanArray)

_, stemLines, _ = plt.stem(range(0,256), histCameraman)
plt.title('Histogram of the image: Cameraman')
plt.xlim(0, 255)
plt.setp(stemLines, 'linewidth', 0.5)
plt.savefig(str(imageRootPath + fileSep + 'histCameraman_orig.jpg'))
plt.show()

## Q4 Match histogram of Lisa to Cameraman
# Perform histogram matching and create the gray scale image from matched array
imgLisaArray_matched = np.array(matchHistogram(imgLisaArray, histCameraman), dtype=np.uint8)
imgLisa_matched = Image.fromarray(imgLisaArray_matched, mode='L')
# Plot new image
plt.imshow(imgLisa_matched, cmap='gray')
plt.axis('off')
plt.title('Histogram matched image: Lisa matched to Cameraman.')
plt.imsave(str(imageRootPath + fileSep + 'imgLisa_matched.jpg'), imgLisa_matched, cmap='gray')
plt.show()
# Obtain histogram of the new image
histLisa_matched = getImagePDF(imgLisaArray_matched)
# Plot the PDF (histogram) of new image
_, stemLines, _ = plt.stem(range(0,256), histLisa_matched)
plt.title('Histogram of the image after matching: Lisa matched to Cameraman')
plt.xlim(0, 255)
plt.setp(stemLines, 'linewidth', 0.5)
plt.savefig(str(imageRootPath + fileSep + 'histLisa_matched.jpg'))
plt.show()

## Q5 Match histogram of Cameraman to Lisa
imgCameramanArray_matched = np.array(matchHistogram(imgCameramanArray, histLisa), dtype=np.uint8)
imgCameraman_matched = Image.fromarray(imgCameramanArray_matched, mode='L')

plt.imshow(imgCameraman_matched, cmap='gray')
plt.axis('off')
plt.title('Histogram matched image: Cameraman matched to Lisa.')
plt.imsave(str(imageRootPath + fileSep + 'imgCameraman_matched.jpg'), imgCameraman_matched, cmap='gray')
plt.show()

histCameraman_matched = getImagePDF(imgCameramanArray_matched)

_, stemLines, _ = plt.stem(range(0,256), histCameraman_matched)
plt.title('Histogram of the image after matching: Cameraman matched to Lisa')
plt.xlim(0, 255)
plt.setp(stemLines, 'linewidth', 0.5)
plt.savefig(str(imageRootPath + fileSep + 'histCameraman_matched.jpg'))
plt.show()

## Q6 Equalize the histogram of Lisa
# Create uniform pdf for equalization
uniformPdf = np.ones((256)) / 256
# Plot the PDF (histogram) of new image
_, stemLines, _ = plt.stem(range(0,256), uniformPdf)
plt.title('Uniform PDF')
plt.xlim(0, 255)
plt.setp(stemLines, 'linewidth', 0.5)
plt.show()

imgLisaArray_equalized = np.array(matchHistogram(imgLisaArray, uniformPdf), dtype=np.uint8)
imgLisa_equalized = Image.fromarray(imgLisaArray_equalized, mode='L')

plt.imshow(imgLisa_equalized, cmap='gray')
plt.axis('off')
plt.title('Histogram equalized image: Lisa.')
plt.imsave(str(imageRootPath + fileSep + 'imgLisa_equalized.jpg'), imgLisa_equalized, cmap='gray')
plt.show()
# Obtain histogram of the new image
histLisa_equalized = getImagePDF(imgLisaArray_equalized)
# Plot the PDF (histogram) of new image
_, stemLines, _ = plt.stem(range(0,256), histLisa_equalized)
plt.title('Histogram of the image after histogram equalization: Lisa')
plt.xlim(0, 255)
plt.setp(stemLines, 'linewidth', 0.5)
plt.savefig(str(imageRootPath + fileSep + 'histLisa_equalized.jpg'))
plt.show()

## Q7 Equalize the histogram of Cameraman
imgCameramanArray_equalized = np.array(matchHistogram(imgCameramanArray, uniformPdf), dtype=np.uint8)
imgCameraman_equalized = Image.fromarray(imgCameramanArray_equalized, mode='L')

plt.imshow(imgCameraman_equalized, cmap='gray')
plt.axis('off')
plt.title('Histogram equalized image: Cameraman.')
plt.imsave(str(imageRootPath + fileSep + 'imgCameraman_equalized.jpg'), imgCameraman_equalized, cmap='gray')
plt.show()

histCameraman_equalized = getImagePDF(imgCameramanArray_equalized)

_, stemLines, _ = plt.stem(range(0,256), histCameraman_equalized)
plt.title('Histogram of the image after histogram equalization: Cameraman')
plt.xlim(0, 255)
plt.setp(stemLines, 'linewidth', 0.5)
plt.savefig(str(imageRootPath + fileSep + 'histCameraman_equalized.jpg'))
plt.show()
