from pylab import *
from PIL import Image
from scipy.ndimage import measurements,morphology

# make sure the image to be grey
image=array(Image.open('/home/charlie/Pictures/Wallpapers/vim.png').convert('L'))
# make sure the image to be binary
image=1*(image<128)
# imshow(image)

# determine the number of objects in 'image'
labels,nbr_objects=measurements.label(image)
print('Number of objects: ',nbr_objects)

# use morphology to be more effective in target separation
im_open=morphology.binary_opening(image,ones((9,5)),iterations=2)
# determine the number of objects in 'image'
labels_open,nbr_objects_open=measurements.label(im_open)
print('Number of objects: ',nbr_objects_open)
