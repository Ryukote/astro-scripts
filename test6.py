import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def mass_temperature_mapping(image_path, output_path='mass_temp_map.png'):
    # Učitaj kao grayscale (intenzitet = aproksimacija mase)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Ne mogu učitati sliku: {image_path}")
    
    # Normalizacija: 0 → hladno (plavo), 1 → toplo (crveno)
    norm = cv2.normalize(img.astype('float32'), None, 0, 1.0, cv2.NORM_MINMAX)

    # Definiraj obrnutu "thermal" colormap (svijetlo = hladno, tamno = toplo)
    colors = [
        (0.0, "aqua"),     # hladno, niska masa
        (0.3, "blue"),
        (0.6, "purple"),
        (0.8, "orange"),
        (1.0, "red"),      # toplo, visoka masa
    ]
    cmap = LinearSegmentedColormap.from_list("mass_temp", colors)

    # Prikaži i spremi rezultat
    plt.imshow(norm, cmap=cmap)
    plt.axis('off')
    plt.title('Distribucija mase / temperature')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.show()

# --- PRIMJER POZIVA ---
mass_temperature_mapping('andromeda.jpg')
