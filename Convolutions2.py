#!/usr/bin/python3
# -*- coding: latin-1 -*-
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#Lecture image en niveau de gris et conversion en float64
img = np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))

(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

#Methode directe
t1 = cv2.getTickCount()
img1 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
imgf = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
module = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
orientation = np.zeros(img.shape)
#sigma = 1

#kernelGaussf = (1/864)*(np.array([[11,23,29,23,11],
#                                  [23,48,62,48,23],
#                                  [29,62,80,62,29],
#                                  [23,48,62,48,23],
#                                  [11,23,29,23,11]]))
#imgf = cv2.filter2D(img,-1,kernel)

for y in range(1,h-1):
  for x in range(1,w-1):
    val = 5*img[y, x] - img[y-1, x] - img[y, x-1] - img[y+1, x] - img[y, x+1] 
    imgf[y,x] = min(max(val,0),255)

for y in range(1,h-1):
  for x in range(1,w-1):
    #img1[y,x] = sigma*(- img[y-1, x-1]  + img[y-1, x+1] - 2*img[y, x-1] + 2*img[y, x+1] - img[y+1, x-1] + img[y+1, x+1])
    #img2[y,x] = sigma*(- img[y-1, x-1]  - 2*img[y-1, x] - img[y-1, x+1] + img[y+1, x-1] + 2*img[y+1, x] + img[y+1, x+1])
    img1[y,x] = - imgf[y, x-1] + imgf[y, x+1]
    img2[y,x] = - imgf[y-1, x] + imgf[y+1, x]
    module[y,x] = np.sqrt(img1[y,x]**2+img2[y,x]**2)
    if img1[y,x] == 0:
      orientation[y,x] = math.pi/2
    else:
      orientation[y,x] = math.atan(img2[y,x]/img1[y,x])

#print(np.max(img1))
#print(np.min(img1))
scaler = MinMaxScaler(feature_range=(0,255))
img1scale = scaler.fit_transform(img1)
img2scale = scaler.fit_transform(img2)
#print(np.max(img1scale))
#print(np.min(img1scale))

modScale = scaler.fit_transform(module)
oriScale = scaler.fit_transform(orientation)

t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Methode directe :",time,"s")

plt.subplot(231)
plt.imshow(img,cmap = 'gray')
plt.title('Originale')

plt.subplot(232)
plt.imshow(img1scale,cmap = 'gray')
plt.title('Ix')

plt.subplot(233)
plt.imshow(img2scale,cmap = 'gray')
plt.title('Iy')

plt.subplot(234)
plt.imshow(module,cmap = 'gray')
plt.title('Module')

plt.subplot(235)
plt.imshow(oriScale,cmap = 'gray')
plt.title('Orientation')



#Methode filter2D
#t1 = cv2.getTickCount()
#kerneldx = np.array([[0, 0, 0],[-1, 0, 1],[0, 0, 0]])
#kerneldy = np.array([[0, -1, 0],[0, 0, 0],[0, 1, 0]])
#kernelSobelDx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
#kernelSobelDy = np.transpose(kernelSobelDx)
#img3 = cv2.filter2D(img,-1,kerneldx)
#img4 = cv2.filter2D(img,-1,kerneldy)
#t2 = cv2.getTickCount()
#time = (t2 - t1)/ cv2.getTickFrequency()
#print("Methode filter2D :",time,"s")

#plt.subplot(232)
#plt.imshow(img3,cmap = 'gray')
#plt.title('dx - filter2d')

#plt.subplot(234)
#plt.imshow(img4,cmap = 'gray')
#plt.title('dy = filter2d')

plt.show()
