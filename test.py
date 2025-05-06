import cv2
import numpy as np
from matplotlib import pyplot as plt

# Učitaj sliku i pretvori u grayscale
img = cv2.imread('andromeda.jpg', cv2.IMREAD_GRAYSCALE)

# Gaussian blur za detekciju "halo komponenti"
blur = cv2.GaussianBlur(img, (51, 51), 0)
halo = cv2.subtract(img, blur)

# CLAHE za pojačanje slabih dijelova
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
halo_enhanced = clahe.apply(halo)

# Prikaži rezultat
plt.imshow(halo_enhanced, cmap='inferno')
plt.title('Halo vizualizacija')
plt.axis('off')
plt.show()
