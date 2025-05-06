import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_halpha(image_path, output_path='halpha_only.png', red_threshold=150, rg_diff=50):
    # Učitaj sliku u BGR formatu
    img = cv2.imread(image_path)

    # Napravi masku gdje je crvena dominantna
    b, g, r = cv2.split(img)
    halpha_mask = (r > red_threshold) & (r - g > rg_diff) & (r - b > rg_diff)

    # Inicijalno sve bijelo
    output = np.ones_like(img) * 255

    # Pikseli sa Hα signalom postaju crveni
    output[halpha_mask] = [0, 0, 255]  # BGR: crvena

    # Spremi i prikaži
    cv2.imwrite(output_path, output)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Hα signal (crveno)")
    plt.show()

# --- PRIMJER POZIVA ---
extract_halpha('andromeda.jpg')
