import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ------------------ Konfiguration ------------------
IMAGE_PATH = "Test_Bild01.jpg"
CHECKPOINT = "sam_vit_b_01ec64.pth"  # kleineres Modell für CPU
MODEL_TYPE = "vit_b"

device = "cpu"
print("Device:", device)

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    crop_n_layers=0,
    min_mask_region_area=5000
)

img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
out = img_bgr.copy()

masks = mask_generator.generate(img_rgb)
print(f"Anzahl Masken: {len(masks)}")

if len(masks) < 2:
    print("Es gibt weniger als 2 Masken, daher wird die größte verwendet.")
    chosen_mask = max(masks, key=lambda x: x['area'])
else:
    masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
    chosen_mask = masks_sorted[1]

seg = chosen_mask['segmentation'].astype(np.uint8) * 255
contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    cv2.drawContours(out, [approx], -1, (0,0,255), 3) #rot
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(out, (x,y),(x+w,y+h), (0,255,0), 2) #grün

# Ergebnis anzeigen und speichern
cv2.imshow("Zweitgrößte Maske", out)
cv2.imwrite("Test_Bild01_sam_second_mask.jpg", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
