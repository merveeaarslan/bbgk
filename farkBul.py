
#14253006 MERVE ARSLAN
#15253016 EDA ÇELTÜK

import cv2
import imutils
from skimage.measure import compare_ssim

resim_1 = cv2.imread("ogr1.jpg")
resim_2 = cv2.imread("ogr2.jpg")

resim_1_boyut = cv2.resize(resim_1, (300, 200))
resim_2_boyut = cv2.resize(resim_2, (300, 200))

resim_1_gri = cv2.cvtColor(resim_1_boyut, cv2.COLOR_BGR2GRAY)
resim_2_gri = cv2.cvtColor(resim_2_boyut, cv2.COLOR_BGR2GRAY)

#diff,iki resim arasındaki farklılıkları bulmak için
(score, diff) = compare_ssim(resim_1_gri, resim_2_gri, full=True)
diff = (diff * 255).astype("uint8")

print("Benzerlik: {}".format(score))
cv2.imshow('diff',diff)
thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#kenarlı resimdeki kontürleri bul sadece en büyüğünü tut.
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for c in cnts:

    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(resim_1_boyut, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(resim_2_boyut, (x, y), (x + w, y + h), (0, 0, 255), 2)
# show the output images
cv2.imshow("resim1",resim_1_boyut)
cv2.imshow("resim2",resim_2_boyut)
cv2.waitKey(0)