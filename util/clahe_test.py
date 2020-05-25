import numpy as np
import cv2


img = cv2.imread('/home/vq218944/MSAI/Low-Light-Enhancement/EnlightenGAN/checkpoints/car_object_trained/web/images/epoch188_real_A.png')
# create a CLAHE object (Arguments are optional).
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

cv2.imwrite('clahe_result.jpg',img)