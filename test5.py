import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_colored_signal(image_path, output_path='rgb_signal_only.png', sigma_thresh=3, background='black'):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Greška: Ne mogu učitati sliku '{image_path}'.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detekcija signala: mean + sigma * std
    mean = np.mean(gray)
    std = np.std(gray)
    threshold = mean + sigma_thresh * std

    # Maska gdje je signal iznad praga
    signal_mask = gray > threshold

    # Kreiraj praznu RGB sliku
    if background == 'black':
        result = np.zeros_like(img)  # sve crno
    elif background == 'white':
        result = np.ones_like(img) * 255  # sve bijelo
    else:
        raise ValueError("background mora biti 'black' ili 'white'.")

    # Sačuvaj boju samo tamo gdje ima signala
    result[signal_mask] = img[signal_mask]

    # Spremi i prikaži
    cv2.imwrite(output_path, result)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'RGB signal only (>{sigma_thresh}σ)')
    plt.show()

# --- PRIMJER ---
extract_colored_signal('andromeda.jpg', background='black')
