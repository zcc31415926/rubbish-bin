import cv2
import sys


img_path = sys.argv[1]
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
mask = img.mean(axis=2) > 200
img[..., -1][mask] = 0
img[..., 0 : 3] = 0
cv2.imwrite('output.png', img)

