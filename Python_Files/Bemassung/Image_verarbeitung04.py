import cv2
import numpy as np

IMAGE_PATH = "Test_Bild01.jpg"

img = cv2.imread(IMAGE_PATH)

# Aufspalten in B, G, R
channels = cv2.split(img)

# Kanten für jeden Kanal
edges = [cv2.Canny(ch, 50, 150) for ch in channels]

# Zusammenführen: Pixel ist Kante, wenn in einem Kanal eine Kante gefunden wird
edges_combined = cv2.bitwise_or(edges[0], edges[1])
edges_combined = cv2.bitwise_or(edges_combined, edges[2])

cv2.imshow("Kanten aus allen Farbkanälen", edges_combined)
cv2.imwrite("edges_color_channels.jpg", edges_combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
