import numpy as np
import cv2


img_path = '/home/charlie/Pictures/michael_jackson.jpg'

# load an color image in grayscale
# parameter 1: any transparency of image will be neglected
# parameter 0: loads image in grayscale mode
# parameter -1: loads image as such including alpha channel
img = cv2.imread(img_path, 0)

# even if the image path is wrong
# it won't throw any error
# but 'print img' will give you none

# display an image
cv2.imshow(img_path, img)
cv2.waitKey(0)
# if 0 is passed, it waits indefinitely for a key stroke
cv2.destroyAllWindows()

# write an image
cv2.imwrite('sample.png', img)

# specify the key stroke in waitKey()
k = cv2.waitKey(0) &0xFF
# wait for the 's' key
if k == ord('s'):
    print('save')
