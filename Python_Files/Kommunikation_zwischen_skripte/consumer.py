import cv2
import numpy as np
from multiprocessing import shared_memory

# Namen und Shapes der Bilder vom Producer-Skript übernehmen
shared_memory_info = [
    {"name": "NameVomShm0", "shape": (480, 640, 3)},  # Beispielwerte
    {"name": "NameVomShm1", "shape": (480, 640, 3)},
    {"name": "NameVomShm2", "shape": (480, 640, 3)},
    {"name": "NameVomShm3", "shape": (480, 640, 3)},
]

for info in shared_memory_info:
    shm = shared_memory.SharedMemory(name=info["name"])
    img = np.ndarray(info["shape"], dtype=np.uint8, buffer=shm.buf)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    _, thresh = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0,0,255), 1)

    cv2.imshow(f"Bearbeitetes Bild {info['name']}", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
