import cv2
import numpy as np
import matplotlib.pyplot as plt

def custom_edge_detector(image):
    #kernal to slide over image
    kernal_x = np.array([[-1, 0, -1],
                       [-2, 0, -2],
                       [-1, 0, -1]])

    kernal_y = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

    # Convolve the image with the Sobel operators
    edges_x = cv2.filter2D(image, -1, kernal_x)
    edges_y = cv2.filter2D(image, -1, kernal_y)

    # Combine horizontal and vertical edges
    edges_combined = np.sqrt(edges_x**2 + edges_y**2)

    # Normalize values to be in the range [0, 255]
    edges_combined = ((edges_combined / edges_combined.max()) * 255).astype(np.uint8)

    return edges_combined
"""
edges = custom_edge_detector(img_gray)

# original and the detected edges
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detector')

plt.show()"""