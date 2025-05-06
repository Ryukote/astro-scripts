import cv2
import numpy as np
import matplotlib.pyplot as plt

def estimate_temperature(rgb_color):
    r, g, b = rgb_color
    if r > 200 and g > 200 and b > 200:
        return 6000
    elif b > r and b > g:
        return 10000
    elif r > b and g < 100:
        return 3500
    elif r > b and g > b:
        return 5000
    else:
        return 6000

def is_near_center(x, y, center_x, center_y, radius=50):
    return (x - center_x)**2 + (y - center_y)**2 < radius**2

def annotate_stars(image_path, output_path='annotated.png'):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- DETEKCIJA CENTRA GALAKSIJE ---
    largest_cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_cnt)
    if M["m00"] != 0:
        galaxy_cx = int(M["m10"] / M["m00"])
        galaxy_cy = int(M["m01"] / M["m00"])
    else:
        galaxy_cx, galaxy_cy = img.shape[1] // 2, img.shape[0] // 2  # fallback

    annotated = img.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

        if w < 2 or h < 2 or w > 20 or h > 20:
            continue  # ignoriši prevelike/male objekte

        if is_near_center(cx, cy, galaxy_cx, galaxy_cy, radius=60):
            continue  # ignoriši objekte blizu centra galaksije

        star_region = img[y:y+h, x:x+w]
        avg_color = cv2.mean(star_region)[:3]
        temperature = estimate_temperature(avg_color)

        cv2.putText(annotated, f"{temperature}K", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 0, 255), 1)

    # Oznaka centra galaksije (za provjeru)
    cv2.circle(annotated, (galaxy_cx, galaxy_cy), 60, (0, 255, 255), 2)

    cv2.imwrite(output_path, annotated)
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Zvijezde s temperaturom (galaksija ignorirana)")
    plt.show()

# --- PRIMJER ---
annotate_stars('andromeda.jpg')
