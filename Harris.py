import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")
print("Type de l'image :",img.dtype)

#Début du calcul
t1 = cv2.getTickCount()
#Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
Theta = np.zeros(img.shape)
# Mettre ici le calcul de la fonction d'intérêt de Harris

didx = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
didy = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
alpha = 0.10
k = 1

for y in range(1,h-1):
  for x in range(1,w-1):
    didx[y,x] =  - 1*img[y-1, x-1]  + 1*img[y-1, x+1] - 2*img[y, x-1] \
      + 2*img[y, x+1] - 1*img[y+1, x-1] + 1*img[y+1, x+1]
    didy[y,x] =  - 1*img[y-1, x-1]  - 2*img[y-1, x] - 1*img[y-1, x+1] \
      + 1*img[y+1, x-1] + 2*img[y+1, x] + 1*img[y+1, x+1]


Ixx = didx ** 2
Iyy = didy ** 2
Ixy = didy * didx

for y in range(k,h-k):
  for x in range(k,w-k):
    Hxx = 0
    Hyy = 0
    Hxy = 0
    for yy in range(y-k,y+k):
        for xx in range (x-k,x+k):
            Hxx += Ixx[yy,xx]
            Hyy += Iyy[yy,xx]
            Hxy += Ixy[yy,xx]

    det = Hxx* Hyy - Hxy * Hxy
    trace = Hxx +Hyy
    Theta[y,x] = det - alpha * trace **2

# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)
#Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0
#On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Mon calcul des points de Harris :",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")

plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
#Relecture image pour affichage couleur
Img_pts=cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"canaux")
print("Type de l'image :",Img_pts.dtype)
#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

plt.show()
