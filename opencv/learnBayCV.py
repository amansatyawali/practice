import numpy as np
import cv2

# img = np.zeros((512, 512, 3), np.uint8)

# cv2.line(img, (0, 0), (511, 511), (255, 150, 255), 5)
# cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 255), 5)
# cv2.circle(img, (50, 50), 150, (255, 255, 0), 3)

# pts = np.array([[425, 40], [440, 60], [460, 20]], np.int32)


# pts2 = np.array([[425, 140], [440, 160], [460, 120]], np.int32)

# cv2.polylines(img, [pts], True, (255, 255, 0))
# cv2.polylines(img, [pts2], False, (0, 255, 0))


# cv2.imshow('sample_image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# img = cv2.imread('eifel_tower.jpg', 1)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


img = cv2.imread('landscape.jpg')
img[ : , : , 0] = 0
img[ : , : , 2] = 0
cv2.imshow('landscape', img)
cv2.waitKey(0)
cv2.destroyAllWindows()