import numpy as np
import cv2
import matplotlib.pyplot as plt
from conv import conv
def Gaussian_filter(sigma = 2):
    x, y = np.mgrid[-1:2, -1:2]
    kernel = (1 / (2 * np.pi * sigma**2 ) ) * np.exp( -(x**2 + y**2) / (2 * sigma**2) )
    kernel = kernel / kernel.sum()
    return kernel


def sobel(img):
    Sx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
    Sy = Sx.T

    Ix = conv(img, Sx)
    Iy = conv(img, Sy)
    angle = np.arctan2(Ix, Iy)
    Ig = np.hypot(Ix, Iy)
    Ig = (Ig / Ig.max() *255).astype("uint8") #轉換成八位元
    print(angle.shape)
    print(Ig.shape)
    #plt.imshow(Ig, cmap = 'gray')
    #plt.show()

    return Ig, angle



def NMS(img, angle):
    #print("NMS shape", img.shape)
    new_contour = np.zeros(img.shape)
    angle = angle * 180 / np.pi #弧度轉角度
    angle[angle < 0] +=180

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            neighbor1 = 255
            neighbor2 = 255

            #angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                neighbor1 = img[i, j+1]
                neighbor2 = img[i, j-1]
            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                neighbor1 = img[i+1, j-1]
                neighbor2 = img[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                neighbor1 = img[i+1, j]
                neighbor2 = img[i-1, j]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                neighbor1 = img[i-1, j-1]
                neighbor2 = img[i+1, j+1]

            if (img[i,j] >= neighbor1) and (img[i,j] >= neighbor2):
                new_contour[i,j] = img[i,j]
            else:
                new_contour[i,j] = 0

    plt.imshow(new_contour, cmap = 'gray')
    plt.show()
    return new_contour



def doublethreshold(img, lowthreshold = 15, highthreshold =30, strong = 255, weak = 25):

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= highthreshold:
                img[i][j] = strong
            elif (img[i][j] < highthreshold and img[i][j] > lowthreshold):
                img[i][j] = weak
    
    return img



def hysteresis(img, weak = 25, strong=255):

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if (img[i,j] == weak):
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                    or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0

    return img


img = cv2.imread("HW2/LBJ.jpg", cv2.COLOR_BGR2GRAY)
#print(bear)
img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
gaussian_img = conv(img, Gaussian_filter())
sobel_img = (sobel(img))
sobelwithg = (sobel(gaussian_img))
x = NMS(sobel_img[0] , sobel_img [1])
y = NMS(sobelwithg[0], sobelwithg [1])
edge_img = hysteresis(doublethreshold(x))
edgewithg = hysteresis(doublethreshold(y))
fig, ax = plt.subplots(2, 2, figsize=(15,8))
ax[0,0].set_title('Sobel With Gaussian')
ax[0,0].imshow(sobel(gaussian_img)[0], cmap='gray')
ax[0,1].set_title('Sobel without Gaussian')
ax[0,1].imshow(sobel_img[0], cmap='gray')
ax[1,0].set_title('Edge With Gaussian')
ax[1,0].imshow(edgewithg, cmap='gray')
ax[1,1].set_title('Edge Without Gaussian')
ax[1,1].imshow(edge_img, cmap='gray')
#plt.savefig("LBJ.png")
plt.show()


fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,8))
blurred = cv2.GaussianBlur(img, (3, 3), 2)
cv2canny = cv2.Canny(blurred, 25, 75)
ax1.set_title('My Canny')
ax1.imshow(edgewithg, cmap='gray')
ax2.set_title('cv2 Canny')
ax2.imshow(cv2canny, cmap='gray')
plt.savefig("CannycompareLBJ2.png")
plt.show()