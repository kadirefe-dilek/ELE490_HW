
import cv2 as cv
import numpy as np


# Define path and name to read the image
imageRootPath = "images"
imageName = "rgbPalette.png"
separator = "\\"

imagePath = str(imageRootPath + separator + imageName)

dim = int(255)
colorPalette = np.zeros((dim,dim,3), dtype=np.uint8)

colorPalette[:,            0:int(dim/3)  , 0] = np.ones((dim,1),dtype=np.uint8) * np.array(range(0,255,int(3*255/dim)))
colorPalette[:,   int(dim/3):int(2*dim/3), 1] = np.ones((dim,1),dtype=np.uint8) * np.array(range(0,255,int(3*255/dim)))
colorPalette[:, int(2*dim/3):int(dim),     2] = np.ones((dim,1),dtype=np.uint8) * np.array(range(0,255,int(3*255/dim)))

cv.imwrite(imagePath, colorPalette)
