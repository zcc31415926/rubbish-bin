from PIL import Image
from pylab import *


img_path = '/home/charlie/Pictures/ubuntu.jpeg'


# image = array(Image.open(img_path).convert('RGB'))
image = array(Image.open(img_path).convert('L'))
# without the color-converting process
# image = array(Image.open('/home/charlie/Pictures/Wallpapers/vim.png'))

# display the image
imshow(image)

# a list of points in the image
x = [100, 100, 400, 400]
y = [200, 500, 200, 500]
# display the points in red & star format
plot(x, y, 'r*')
# draw a line between (x[0], y[0]) and (x[1], y[1])
plot(x[:2], y[:2])
# add title
title('Plotting')
# show the image
show()

# create an image
figure()
# ignore color information
gray()
# display the frame at the left-top corner
contour(image, origin='image')
axis('equal')
axis('off')

# use hist() to display histograms
figure()
hist(image.flatten(), 128)
show()
