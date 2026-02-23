#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import os
import time
import json

CALIB_FILE = "distanz_calibration.json"

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = True           # für genauere Tiefe empfehle ich True
# Better handling for occlusions:
lr_check = True

# Ordner für gespeicherte Bilder anlegen
SAVE_DIR = "saved_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# ROI-Größe (quadratisch, zentriert)
ROI_SIZE = 100  # Pixel

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

# Ausgänge: Disparity (für Farbe) und Depth (für Meter)
xout_disparity = pipeline.create(dai.node.XLinkOut)
xout_disparity.setStreamName("disparity")
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")        # Veraltet, aber funktioniert noch
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# Stereo-Konfiguration
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)  # statt HIGH_DENSITY
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(lr_check)
stereo.setExtendedDisparity(extended_disparity)
stereo.setSubpixel(subpixel)

# Feinjustierung der Filter
config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 100   # 100 mm = 0.1 m
config.postProcessing.thresholdFilter.maxRange = 1000  # 1000 mm = 1.0 m
config.postProcessing.decimationFilter.decimationFactor = 1
stereo.initialConfig.set(config)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.disparity.link(xout_disparity.input)
stereo.depth.link(xout_depth.input)          # NEU: Tiefen-Stream

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues
    q_disparity = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    print("Live-Disparitätsanzeige mit zentriertem ROI gestartet.")
    print("Drücke 's' um das aktuelle Bild zu speichern, 'q' zum Beenden.")

    while True:
        # Neueste Frames holen
        in_disparity = q_disparity.get()
        in_depth = q_depth.get()

        disparity_frame = in_disparity.getFrame()
        depth_frame = in_depth.getFrame()   # in Metern (float32)

        # Disparity für Anzeige normalisieren
        max_disp = stereo.initialConfig.getMaxDisparity()
        disp_norm = (disparity_frame * (255.0 / max_disp)).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

        # Bildmitte und ROI-Koordinaten berechnen
        h, w = disparity_frame.shape
        cx, cy = w // 2, h // 2
        half = ROI_SIZE // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        # ROI aus dem Tiefenbild extrahieren
        roi_depth = depth_frame[y1:y2, x1:x2]

        # Gültige Tiefenwerte (z.B. > 0.1 m und < 10 m) filtern
        valid = roi_depth[roi_depth > 0]   # ignoriert nur 0
        if valid.size > 0:
            min_z = np.min(valid)
        else:
            min_z = float('nan')

        # ROI im Farbbild einzeichnen (grünes Rechteck)
        cv2.rectangle(disp_color, (x1, y1), (x2, y2), (0, 255, 0), 2)


        hoehe = float('nan')          # Standardwert
        if os.path.exists(CALIB_FILE):
            with open(CALIB_FILE, "r") as f:
                calib_data = json.load(f)
            z_mean = calib_data.get("z_median", float('nan'))
            hoehe = z_mean - min_z

        # Jetzt ist hoehe immer definiert (ggf. nan)
        text = f"hoehe: {hoehe:.1f} mm" if not np.isnan(hoehe) else "hoehe: ---"
        text = f"z_min: {min_z:.1f} mm" if not np.isnan(min_z) else "z_min: ---"

        cv2.putText(disp_color, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Anzeigen
        cv2.imshow("disparity (grau)", disp_norm)
        cv2.imshow("disparity (farbe) mit ROI", disp_color)

        # Tastendruck
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            filename = os.path.join(SAVE_DIR, f"disparity_color_{timestamp}.png")
            cv2.imwrite(filename, disp_color)
            print(f"Bild gespeichert: {filename}")

    cv2.destroyAllWindows()