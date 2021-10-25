**Name : Samiksha Surawashi**

**Task 4 : Image to Pencil Sketch with Python**
"""

from google.colab.patches import cv2_imshow

import cv2

image = cv2.imread("/content/Hyderabad_pic.jpg")

img = cv2.resize(image,(600,500))

cv2_imshow(img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(img_gray)

img_invert = cv2.bitwise_not(img_gray)
cv2_imshow(img_invert)

img_smoothing = cv2.GaussianBlur(img_invert, (21,21),sigmaX=0, sigmaY=0)
cv2_imshow(img_smoothing)

def dodgeV2(x, y):
  return cv2.divide(x, 255 - y, scale=256)

final_img = dodgeV2(img_gray, img_smoothing)
cv2_imshow(final_img)

cv2_imshow(img)
cv2_imshow(final_img)

"""**THANK YOU**"""
