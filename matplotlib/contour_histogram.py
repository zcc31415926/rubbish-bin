from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


img_path = 'img.jpg'

image_rgb = np.array(Image.open(img_path).convert('RGB'))
image_gray = np.array(Image.open(img_path).convert('L'))

# the original image
plt.title('Image')
plt.imshow(image_rgb)

# the contour
plt.figure()
plt.contour(image_gray, origin='image')
plt.title('Contour')
plt.axis('off')

# the histogram
plt.figure()
plt.hist(image_rgb.flatten(), 128)
plt.title('Histogram')
plt.show()

