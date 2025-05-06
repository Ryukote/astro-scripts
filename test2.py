import numpy as np
import cv2
import matplotlib.pyplot as plt
from astropy.io import fits
import os

def load_image(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img.astype(np.float32)
    elif ext == '.fits':
        with fits.open(path) as hdul:
            data = hdul[0].data
            if data.ndim > 2:
                data = data[0]
            return np.nan_to_num(data.astype(np.float32))
    else:
        raise ValueError("Nepodržani format slike.")

def detect_signal(image, sigma_threshold=3.0):
    background = np.median(image)
    std = np.std(image)
    threshold = background + sigma_threshold * std
    signal_mask = image >= threshold
    return signal_mask

def create_output_image(signal_mask):
    h, w = signal_mask.shape
    output = np.ones((h, w, 3), dtype=np.uint8) * 255  # bijela pozadina

    # Ljubičasta boja: RGB = (128, 0, 128)
    output[signal_mask] = [128, 0, 128]
    return output

def process_astrophoto(path, output_path=None):
    image = load_image(path)
    signal_mask = detect_signal(image)
    output_image = create_output_image(signal_mask)

    # Prikaz rezultata
    plt.imshow(output_image)
    plt.axis('off')
    plt.title("3σ Signal Mask")
    plt.show()

    # Spremanje rezultata
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        print(f"Rezultat spremljen u: {output_path}")

# --- PRIMJER POZIVA ---
# Promijeni 'astro.jpg' u svoj fajl
process_astrophoto('andromeda.jpg', 'output.png')
