import numpy as np
import cv2
import matplotlib.pyplot as plt
from conv import conv
def Gaussian_filter(sigma = 5):
    x, y = np.mgrid[-1:2, -1:2]
    kernel = (1 / (2 * np.pi * sigma**2 ) ) * np.exp( -(x**2 + y**2) / (2 * sigma**2) )
    kernel = kernel / kernel.sum()
    return kernel

bear = cv2.imread("HW2\gray_bear3.png", cv2.COLOR_BGR2GRAY)
bear = conv(bear, Gaussian_filter())
cv2.imwrite("bearblur.png", bear)

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
    plt.imshow(Ig, cmap = 'gray')
    plt.show()

    return Ig, angle

bear = sobel(bear)

def NMS(img, angle):
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

x = NMS(bear[0], bear[1])

def doublethreshold(img, lowratio = 0.05, highratio = 0.1, strong = 255, weak = 55):
    highthreshold = img.max() * highratio
    lowthreshold = highthreshold * lowratio

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= highthreshold:
                img[i][j] = strong
            elif (img[i][j] < highthreshold and img[i][j] > lowthreshold):
                img[i][j] = weak
    
    return img

plt.imshow(doublethreshold(x), cmap='gray')
plt.show()

def hysteresis(img, weak = 55, strong=255):

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

plt.imshow(hysteresis(doublethreshold(x)), cmap='gray')
plt.show()