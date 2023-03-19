import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pylab


num_points = 3

image = np.ones((400, 600, 3)) * 255
plt.imshow(image)
print(f'please click {num_points} points')
points = pylab.ginput(num_points)
plt.figure()
plt.imshow(image)
points = np.array(points).astype(np.int32)
for i in range(num_points):
    print(f'point {i + 1} clicked: {points[i]}')
plt.plot(points[:, 0], points[:, 1], 'r*')
for i in range(1, num_points):
    plt.plot([points[i - 1, 0], points[i, 0]], [points[i - 1, 1], points[i, 1]])
plt.show()

