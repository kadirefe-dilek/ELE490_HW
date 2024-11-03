# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ELE490 - Fundamentals of Image Processing 
# Assignment 4
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

    *Runtime warnings due to division by zero occures when the input image is uniform. To Prevent any warning due to division by zero, warnings module is used to raise "RuntimeWarning" type warnings as errors which is tolerated in except case by assigning a zeros array to the "fullGray".
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


'''
Function to calculate Fc[p,q] (DFT signal)) from the given f[n,m] (space signal)
    in:
        f_nm: Input numpy array (assumed to be 2d) to operate on.
    out:
        fc_pq: Fourier domain signal in 2D found by taking an fft after necessary manipulations are done on f_nm.
'''
def calculateFc_pq(f_nm:np.ndarray):
    fc_pq = np.zeros(f_nm.shape)
    for n in range(f_nm.shape[0]):
        for m in range(f_nm.shape[1]):
            fc_pq[n,m] = (-1)**(n+m) * f_nm[n,m]
    
    fc_pq = np.abs(np.fft.fft2(fc_pq))

    return np.array(fc_pq)


'''
Function that generates an edge image of size 512x512 for the given angle theta.
    in:
        theta: (Assumed to be in radians) Angle that defines the edge. 
    out:
        edgeImage: Output numpy array of size 512x512 with an edge of angle theta (in radians). 
'''
def generateEdgeImage(theta):
    # Create a full white image. 512 can be replaced by an argument 'size'
    edgeImage = np.ones((512, 512)) * 255 
    # Calculate the necessary parameters
    tan_theta = math.tan(theta)
    center_x = 512 // 2
    center_y = 512 // 2
    
    # Decide the edge and fill the image below it
    for y in range(512):
        for x in range(512):
            # Relative coordinates wrt center
            rel_x = x - center_x
            rel_y = y - center_y

            # Decide where to fill and assign 0 (black)
            if theta == 0:
                # If theta=0, then fill the below half
                if y > 512 // 2:
                    edgeImage[y, x] = 0 
            elif theta == math.pi / 2:
                # If theta=pi/2, then fill the left half
                if x < 512 // 2:
                    edgeImage[y, x] = 0 
            else:
                # For different theta, fill below the edge
                if rel_y > rel_x * tan_theta:
                    edgeImage[y, x] = 0 
    
    return edgeImage



## Definitions for reading and saving images
# Define path and name to read the image
fileSep = "\\"
imageRootPath = gcd() 
imageRootPath = imageRootPath + fileSep + "images"


## Q1 w_r = pi/16 and for 0.5*w_r and 2*w_r
# Initial definitions
OMEGA_r = [np.pi/16, np.pi/32, np.pi/8]
THETA = [0, np.pi/6, np.pi/3, np.pi/2]
nameSufOmega = ["wr", "05e-1wr", "2wr"]

for r,omega_r in enumerate(OMEGA_r):
    
    omega_x = omega_r/(np.sqrt(1+np.tan(THETA)))
    omega_y = np.tan(THETA) * omega_x

    F_of_nm = np.zeros((512,512,4))
    for n in range(512):
        for m in range(512):
            F_of_nm[n,m,:] = np.cos(omega_x.reshape((1,1,4))*n + omega_y.reshape((1,1,4))*m) + 1/2

    for i,theta in enumerate(THETA):
        angleDegree = int(round(180*theta/np.pi))
        f_nm = F_of_nm[:,:,i]

        img_f_nm = getFullGrayScale(f_nm)
        # Display
        plt.imshow(img_f_nm, cmap="gray")
        plt.axis("off")
        plt.title(f"f[n,m], theta={angleDegree} degrees, w={nameSufOmega[r]}")
        plt.imsave(str(imageRootPath + fileSep + "img_f_nm_" + str(angleDegree) + "deg_" + nameSufOmega[r] + ".jpg"), img_f_nm, cmap="gray")
        plt.show()


        fd_pq = np.abs(np.fft.fft2(f_nm))
        img_fd_pq = getFullGrayScale(fd_pq)
        # Display
        plt.imshow(img_fd_pq, cmap="gray")
        plt.axis("off")
        plt.title(f"fd[p,q], theta={angleDegree} degrees, w={nameSufOmega[r]}")
        plt.imsave(str(imageRootPath + fileSep + "img_fd_pq_" + str(angleDegree) + "deg_" + nameSufOmega[r] + ".jpg"), img_fd_pq, cmap="gray")
        plt.show()


        fc_pq = calculateFc_pq(f_nm)
        img_fc_pq = getFullGrayScale(fc_pq)
        # Display
        plt.imshow(img_fc_pq, cmap="gray")
        plt.axis("off")
        plt.title(f"fc[p,q], theta={angleDegree} degrees, w={nameSufOmega[r]}")
        plt.imsave(str(imageRootPath + fileSep + "img_fc_pq_" + str(angleDegree) + "deg_" + nameSufOmega[r] + ".jpg"), img_fc_pq, cmap="gray")
        plt.show()

        fl_pq = np.zeros(fc_pq.shape)
        fl_pq = np.log10(fc_pq, out=fl_pq, where=fc_pq > 0)
        img_fl_pq = getFullGrayScale(fl_pq)
        # Display
        plt.imshow(img_fl_pq, cmap="gray")
        plt.axis("off")
        plt.title(f"fl[p,q], theta={angleDegree} degrees, w={nameSufOmega[r]}")
        plt.imsave(str(imageRootPath + fileSep + "img_fl_pq_" + str(angleDegree) + "deg_" + nameSufOmega[r] + ".jpg"), img_fl_pq, cmap="gray")
        plt.show()


## Q2-a
edgeImage45 = generateEdgeImage(math.pi/4)

# Display
plt.imshow(edgeImage45, cmap="gray")
plt.axis("off")
plt.title("Edge with theta=45 degrees")
plt.imsave(str(imageRootPath + fileSep + "img_edgeImage45.jpg"), edgeImage45, cmap="gray")
plt.show()

Ed_pq_45 = abs(np.fft.fft2(edgeImage45))
Ec_pq_45 = calculateFc_pq(edgeImage45)
El_pq_45 = np.log10(Ec_pq_45)

img_Ed_pq_45 = getFullGrayScale(Ed_pq_45)
plt.imshow(img_Ed_pq_45, cmap="gray")
plt.title("Fd[p,q] for 45 degrees edge")
plt.axis("off")
plt.imsave(str(imageRootPath + fileSep + "img_Ed_pq_45.jpg"), img_Ed_pq_45, cmap="gray")
plt.show()

img_Ec_pq_45 = getFullGrayScale(Ec_pq_45)
plt.imshow(img_Ec_pq_45, cmap="gray")
plt.title("Fc[p,q] for 45 degrees edge")
plt.axis("off")
plt.imsave(str(imageRootPath + fileSep + "img_Ec_pq_45.jpg"), img_Ec_pq_45, cmap="gray")
plt.show()

img_El_pq_45 = getFullGrayScale(El_pq_45)
plt.imshow(img_El_pq_45, cmap="gray")
plt.title("Fl[p,q] for 45 degrees edge")
plt.axis("off")
plt.imsave(str(imageRootPath + fileSep + "img_El_pq_45.jpg"), img_El_pq_45, cmap="gray")
plt.show()

## Q2-b Repeat (a) for different theta
for theta in [0, np.pi/6, np.pi/3, np.pi/2]:
    angleDegree = int((theta/np.pi) * 180)

    edgeImage = generateEdgeImage(theta)

    # Display
    plt.imshow(edgeImage, cmap="gray")
    plt.axis("off")
    plt.title(f"Edge with theta={angleDegree} degrees")
    plt.imsave(str(imageRootPath + fileSep + "img_edgeImage_" + str(angleDegree) + ".jpg"), edgeImage, cmap="gray")
    plt.show()
    # initialize F_d/_c/_l[p,q] with zeros arrays
    Ed_pq = np.zeros(edgeImage.shape)
    Ec_pq = np.zeros(edgeImage.shape)
    El_pq = np.zeros(edgeImage.shape)
    # calculate F_d/_c/_l[p,q] functions
    Ed_pq = abs(np.fft.fft2(edgeImage))
    Ec_pq = calculateFc_pq(edgeImage)
    El_pq = np.log10(Ec_pq, out=El_pq, where=Ec_pq > 0)

    img_Ed_pq = getFullGrayScale(Ed_pq)
    plt.imshow(img_Ed_pq, cmap="gray")
    plt.title(f"Fd[p,q] for {angleDegree} degrees edge")
    plt.axis("off")
    plt.imsave(str(imageRootPath + fileSep + "img_Ed_pq_" + str(angleDegree) + ".jpg"), img_Ed_pq, cmap="gray")
    plt.show()

    img_Ec_pq = getFullGrayScale(Ec_pq)
    plt.imshow(img_Ec_pq, cmap="gray")
    plt.title(f"Fc[p,q] for {angleDegree} degrees edge")
    plt.axis("off")
    plt.imsave(str(imageRootPath + fileSep + "img_Ec_pq_" + str(angleDegree) + ".jpg"), img_Ec_pq, cmap="gray")
    plt.show()

    img_El_pq = getFullGrayScale(El_pq)
    plt.imshow(img_El_pq, cmap="gray")
    plt.title(f"Fl[p,q] for {angleDegree} degrees edge")
    plt.axis("off")
    plt.imsave(str(imageRootPath + fileSep + "img_El_pq_" + str(angleDegree) + ".jpg"), img_El_pq, cmap="gray")
    plt.show()
