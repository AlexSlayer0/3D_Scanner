import cv2
import os

# ---------------- Bild laden ----------------
image_path = r"C:\Users\grane\Felix_Schule\Diplomarbeit\Bemassung\Test_Bild02.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Bild konnte nicht geladen werden: {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
grad = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))

_, thresh = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ---------------- Alle Konturen rot zeichnen ----------------
cv2.drawContours(image, contours, -1, (0, 0, 255), 1)  # rot
max_len = 0
longest_contour = None
for c in contours:
    length = cv2.arcLength(c, closed=True)
    if length > max_len:
        max_len = length
        longest_contour = c

# ---------------- Grünes Rechteck um längste Kontur ----------------
if longest_contour is not None:
    x, y, width, height = cv2.boundingRect(longest_contour)
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Horizontale Seite oben
    text_top = f"{width}px"
    text_size_top, _ = cv2.getTextSize(text_top, font, font_scale, thickness)
    text_x_top = x + (width - text_size_top[0]) // 2
    text_y_top = y - 10
    cv2.putText(image, text_top, (text_x_top, text_y_top), font, font_scale, (0, 255, 0), thickness)

    # Vertikale Seite links
    text_left = f"{height}px"
    text_size_left, _ = cv2.getTextSize(text_left, font, font_scale, thickness)
    text_x_left = x - text_size_left[0] - 10
    text_y_left = y + (height + text_size_left[1]) // 2
    cv2.putText(image, text_left, (text_x_left, text_y_left), font, font_scale, (0, 255, 0), thickness)


    print(f"Längste Kante: {max_len:.2f} Pixel")
    print(f"Grünes Rechteck: Breite={width}px, Höhe={height}px")
else:
    print("Keine Konturen gefunden.")


# ---------------- Ergebnis speichern ----------------
output_path = os.path.join(os.path.dirname(image_path), "edges_sobel.jpg")
cv2.imwrite(output_path, image)
print(f"Ergebnis gespeichert unter: {output_path}")



