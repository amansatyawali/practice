import numpy as np
import cv2

# cap = cv2.VideoCapture('C0001.MP4')
cap = cv2.VideoCapture(0)

ret = True
while cap.isOpened() :
  ret, frame = cap.read()
  print(frame.shape)
  frame = frame[ : , : : -1, :]
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  sobelY = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize = 3)
  sobelX = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize = 3)
  # frame = sobelX + sobelY
  if ret :
    cv2.imshow('cam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
      break
  else : 
    break

cap.release()
cv2.destroyAllWindows()