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
module = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
orientation = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)

for y in range(1,h-1):
  for x in range(1,w-1):
    valdx = - 1*img[y-1, x-1]  + 1*img[y-1, x+1] - 2*img[y, x-1] + 2*img[y, x+1] - 1*img[y+1, x-1] + 1*img[y+1, x+1]
    valdy = - 1*img[y-1, x-1]  - 2*img[y-1, x] - 1*img[y-1, x+1] + 1*img[y+1, x-1] + 2*img[y+1, x] + 1*img[y+1, x+1]
    #img2[y,x] = min(max(valdy,0),255)
    #img1[y,x] = min(max(valdx,0),255)
    img2[y,x] = valdy
    img1[y,x] = valdx

scaler = MinMaxScaler(feature_range=(0,255))
img1scale = scaler.fit_transform(img1)
img2scale = scaler.fit_transform(img2)

for y in range(1,h-1):
  for x in range(1,w-1):
    module[y,x] = np.sqrt(img1scale[y,x]**2+img2scale[y,x]**2)
    if img1scale[y,x] == 0:
      orientation[y,x] = math.pi/2
    else:
      orientation[y,x] = math.atan(img2scale[y,x]/img1scale[y,x])

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
plt.imshow(modScale,cmap = 'gray')
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
