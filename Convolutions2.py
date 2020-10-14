#!/usr/bin/python3
# -*- coding: latin-1 -*-
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

# Lecture image en niveau de gris et conversion en float64
img = np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png', 0))

(h, w) = img.shape
print("Dimension de l'image :", h, "lignes x", w, "colonnes")

# Methode directe
t1 = cv2.getTickCount()
img_dx = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
img_dy = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
imgf = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
module = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
orientation = np.zeros(img.shape)

for y in range(1, h-1):
    for x in range(1, w-1):
        img_dx[y, x] = -img[y-1, x-1] +img[y-1, x+1] -2 *img[y, x-1] +2*img[y, x+1] -img[y+1, x-1] +img[y+1, x+1]
        img_dy[y, x] = -img[y-1, x-1] -2*img[y-1, x] -img[y-1,x+1] +img[y+1, x-1] +2*img[y+1, x] +img[y+1, x+1]
        module[y, x] = np.sqrt(img_dx[y, x]**2+img_dy[y, x]**2)
        orientation[y, x] = np.arctan2(img_dy[y, x], img_dx[y, x])

t2 = cv2.getTickCount()
time = (t2 - t1) / cv2.getTickFrequency()
print("Methode directe :", time, "s")

plt.subplot(231)
plt.imshow(img, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('Originale')

plt.subplot(232)
plt.imshow(img_dx, cmap='gray', vmin=-255.0, vmax=255.0)
plt.title('Ix')

plt.subplot(233)
plt.imshow(img_dy, cmap='gray', vmin=-255.0, vmax=255.0)
plt.title('Iy')

plt.subplot(234)
plt.imshow(module, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('Module')

plt.subplot(235)
plt.imshow(orientation, cmap='gray', vmin=-math.pi, vmax=math.pi)
plt.title('Orientation')


# Methode filter2D
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

# plt.subplot(232)
#plt.imshow(img3,cmap = 'gray')
#plt.title('dx - filter2d')

# plt.subplot(234)
#plt.imshow(img4,cmap = 'gray')
#plt.title('dy = filter2d')

plt.show()
