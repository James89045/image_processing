import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def Conv(img, kernel, RGB = False):
    new_row = img.shape[0] - kernel.shape[0] + 1
    new_col = img.shape[1] - kernel.shape[0] + 1
    if RGB == True:
        new_img = np.zeros(((new_row), (new_col), 3))
        for ch in range(3):
            for i in range(int(new_row)):
                for j in range(int(new_col)):
                    value = 0
                    for ki in range(3):
                        for kj in range(3):
                            value += img[:,:,ch][i+ki][j+kj] * kernel[ki][kj]
                            new_img[i][j] = value   
    else:
        new_img = np.zeros((int(new_row), int(new_col)))
        for i in range(int(new_row)):
            for j in range(int(new_col)):
                value = 0
                for ki in range(3):
                    for kj in range(3):
                        value += img[i+ki][j+kj] * kernel[ki][kj]
                        new_img[i][j] = value
    #print(new_img.shape)   
    return new_img.astype('uint8')

def make_histogram(img, RGB = False):
    if RGB == True:
        hist_plot = np.zeros((3,256))
        for ch in range(3):
            his_list = []
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    his_list.append(img[i][j][ch])
            his_list = np.array(his_list)
            for x in range(256):
                hist_plot[ch][x] += sum(his_list == x)
 
        plt.plot(range(256), hist_plot[0], c = 'b', label = 'Blue channel')
        plt.plot(range(256), hist_plot[1], c = 'g', label = 'Green channel')
        plt.plot(range(256), hist_plot[2], c = 'r', label = 'Red channel')
        plt.legend(loc = 'lower right')
        plt.show()
    else:
        his_list = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                his_list.append(img[i][j])
        his_list = np.array(his_list)
        plt.hist(his_list, bins='auto')
        plt.title('histogram')
        plt.show()

    return 0


def hist_norm(img, RGB = False):
    if RGB == True:
        for ch in range(3):
            hist_array = np.bincount(img[:,:,ch].flatten(), minlength=256)
            num_pixel = np.sum(hist_array)
            hist_array = hist_array / num_pixel
            chist_array = np.cumsum(hist_array)
            transform = np.floor(255 * chist_array).astype(np.uint8)
            img_list = list(img[:,:,ch].flatten())
            new_img_ch = [transform[p] for p in img_list]
            new_img_ch = np.array(new_img_ch).reshape(img.shape[0], img.shape[1], 1)

            if ch == 0:
                B_img = new_img_ch            
            if ch == 1:
                G_img = new_img_ch
            if ch == 2:
                R_img = new_img_ch
        new_img = np.concatenate((B_img, G_img, R_img), axis=2).astype('uint8')

    if RGB == False:
        hist_array = np.bincount(img[:,:].flatten(), minlength=256)
        num_pixel = np.sum(hist_array)
        hist_array = hist_array / num_pixel
        chist_array = np.cumsum(hist_array)
        transform = np.floor(255 * chist_array).astype(np.uint8)
        img_list = list(img[:,:].flatten())
        new_img = [transform[p] for p in img_list]
        new_img = np.array(new_img).reshape(img.shape[0], img.shape[1]).astype('uint8')

    return new_img

def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final.astype('uint8')


def mean_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            data_final[i][j] = sum(temp)/len(temp)
            temp = []
    return data_final.astype('uint8')

def gaussian_noise(img, mean=0, sigma=0.3):
    #normalize
    img = img / 255.0
    noise = np.random.normal(mean, sigma, img.shape)
    gaussian_out = img + noise
    gaussian_out = np.clip(gaussian_out, 0, 1)
    gaussian_out = np.uint8(gaussian_out*255)
    noise = np.uint8(noise*255)

    return gaussian_out

def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 255
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


gaussian_filter = 1/16 * np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]])


img = cv2.imread('bear.jpg')
img = cv2.resize(img,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_LINEAR)
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('blackbear.png', img_g)
img_g_n = sp_noise(img_g, 0.1)
img_median = median_filter(img_g_n, 3)
img_mean = mean_filter(img_g_n, 3)
img_gaussian = Conv(img_g_n, gaussian_filter)

fig, ax = plt.subplots(2,2,figsize=(10,6))
ax[0,0].set_title('image with noise')
ax[0,0].imshow(img_g_n, cmap='gray')
ax[0,1].set_title('median filter')
ax[0,1].imshow(img_median, cmap='gray')
ax[1,0].set_title('mean filter')
ax[1,0].imshow(img_mean, cmap='gray')
ax[1,1].set_title('Gaussian filter')
ax[1,1].imshow(img_gaussian, cmap='gray')
plt.show()

#histogram
turtle = cv2.imread('turtle.jpg')
turtle = cv2.resize(turtle,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_LINEAR)
turtle = hist_norm(turtle, True)
cv2.imwrite('norm_turtle.png', turtle)
bear = cv2.imread('bear2.jpg')
bear = cv2.resize(bear,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_LINEAR)
bear = hist_norm(bear, True)
cv2.imwrite('norm_bear.png', bear)
make_histogram(turtle, True)
make_histogram(bear, True)

#實驗3
bear2 = cv2.imread("bear3.jpg")
bear2 = cv2.resize(bear2,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
bear_gray = cv2.cvtColor(bear2, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_bear3.png', bear_gray)
make_histogram(bear2, True)
make_histogram(bear_gray)
norm_bear2 = hist_norm(bear2, True)
norm_bear_gray = hist_norm(bear_gray)
make_histogram(norm_bear2, True)
make_histogram(norm_bear_gray)
cv2.imwrite('norm_bear2.png',norm_bear2)
cv2.imwrite('norm_bear_gray.png', norm_bear_gray)

#實驗4
laplacian = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]])

lap_bear1 = Conv(img_g, laplacian)
cv2.imwrite('lapbear.png', lap_bear1)
lap_bear2 = Conv(bear_gray, laplacian)
cv2.imwrite('lapbear2.png', lap_bear2)