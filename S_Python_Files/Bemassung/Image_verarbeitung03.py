import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import requests
import os

# ----------------- U2NETP Model Definition -----------------
# Minimal Version von U2NET
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# ----------------- Konfiguration -----------------
IMAGE_PATH = "Test_Bild01.jpg"
OUTPUT_PATH = "Test_Bild01_u2netp.jpg"
DEVICE = "cpu"
CHECKPOINT_PATH = "u2netp.pth"  # Pfad zum vortrainierten U2NETP

# ----------------- Bild laden -----------------
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Bild '{IMAGE_PATH}' nicht gefunden.")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_bgr.shape[:2]

# ----------------- Bild vorbereiten -----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

input_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

# ----------------- Modell laden -----------------

mask_dummy = np.zeros((320,320), dtype=np.uint8)
cv2.rectangle(mask_dummy, (50,50), (270,270), 255, -1)  # Dummy rechteckige Maske

# ----------------- Maske auf Originalbild skalieren -----------------
mask_resized = cv2.resize(mask_dummy, (w,h))
mask_bin = (mask_resized > 0).astype(np.uint8) * 255

# ----------------- Konturen finden -----------------
contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    raise RuntimeError("Keine Konturen gefunden.")

# Flächen berechnen und zweitgrößte Kontur wählen
areas = [cv2.contourArea(c) for c in contours]
idx = np.argsort(areas)[-2] if len(areas) >= 2 else np.argmax(areas)
chosen_cnt = contours[idx]

# Rechteck-Umriss approximieren
epsilon = 0.01 * cv2.arcLength(chosen_cnt, True)
approx = cv2.approxPolyDP(chosen_cnt, epsilon, True)

# ----------------- Ergebnis zeichnen -----------------
out = img_bgr.copy()
cv2.drawContours(out, [approx], -1, (0,0,255), 2)  # Polygon-Umriss rot
x, y, w_box, h_box = cv2.boundingRect(chosen_cnt)
cv2.rectangle(out, (x,y), (x+w_box, y+h_box), (0,255,0), 2)  # Bounding Box grün

# ----------------- Ergebnis speichern & anzeigen -----------------
cv2.imwrite(OUTPUT_PATH, out)
cv2.imshow("U2NETP Result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
