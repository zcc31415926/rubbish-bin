import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from PIL import Image
from pylab import *


img_path = '/home/charlie/Pictures/ubuntu.jpeg'
image = array(Image.open(img_path))
imshow(image)
print('please click 3 points')
# the function ginput() in Pylab realizes interaction
x = ginput(3)
print('you clicked:', x)
show()
