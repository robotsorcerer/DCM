# draw a curve. Ref: https://stackoverflow.com/questions/60659978/how-do-i-generate-a-curved-tube-from-2d-slices-by-shifting-center-coordinates
import cv2
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
# Define some points:
points = np.array([[0, 1, 8, 2, 2],
                   [1, 0, 6, 7, 2]]).T  # a (nbre_points x nbre_dim) array

# Linear length along the line:
distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
distance = np.insert(distance, 0, 0)/distance[-1]

alpha = np.linspace(0, 1, 128)

method = 'cubic'   
interpolator =  interp1d(distance, points, kind=method, axis=0)
curve = interpolator(alpha)

print("curve: ", curve, curve.shape)

img = np.zeros((128, 128, 3), np.uint8) + 255
radius = 10

tube_matrix = []
for i in range(128):    
    circle_center = np.round(curve[i]*12 + 15).astype(int)
    slice_2d = cv2.circle(np.zeros((128,128)), tuple(circle_center), radius, color=1, thickness=-1)
    tube_matrix.append(slice_2d)

    #Draw cicle on image - for testing
    img = cv2.circle(img, tuple(circle_center), radius, color=(i*10 % 255, i*20 % 255, i*30 % 255), thickness=2)
print("curve: ", curve[5], curve.shape)

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(121)
ax.imshow(img)

ax = fig.add_subplot(122)
ax.plot(*curve.T, "r*")
ax.set_title("Curves")
plt.show()