import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image

# Beispiel: Koordinaten der Barcode-Ecken im Originalbild (manuell geschätzt)
src_pts = np.float32([
    [x0, y0],  # obere linke Ecke
    [x1, y0],  # obere rechte Ecke
    [x1, y1],  # untere rechte Ecke
    [x0, y1],  # untere linke Ecke
])

# Zielrechteck: gewünschte Größe des Barcodes nach Transformation
width = 300
height = 100
dst_pts = np.float32([
    [0, 0],
    [width-1, 0],
    [width-1, height-1],
    [0, height-1]
])

img = cv2.imread("ProductBarcode401.jpg")
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, M, (width, height))

# Optional: in Graustufen + Vorverarbeitung
gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# Decode
pil_img = Image.fromarray(gray)
barcodes = decode(pil_img)
for b in barcodes:
    print("Typ:", b.type, "Wert:", b.data.decode("utf-8"))
