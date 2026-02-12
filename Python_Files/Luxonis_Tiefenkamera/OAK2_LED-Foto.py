#!/usr/bin/env python3
import depthai as dai
import cv2
import time
import serial
import os
from datetime import datetime

# ===== SERIELLE LED-STEUERUNG =====
def control_light(state: bool):
    port = "/dev/ttyUSB0"
    baudrate = 9600
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            time.sleep(1.5)
            ser.write(b"Change\n")
            time.sleep(0.1)
            ser.write(b"a\n" if state else b"0\n")
            time.sleep(0.2)
        print(f"Licht {'EIN' if state else 'AUS'}")
    except Exception as e:
        print(f"?? Lichtsteuerung: {e}")

# ===== 12 MP STILL STREAM =====
print("\n=== OAK-D2 12 MP STILL TEST ===")
control_light(True)
time.sleep(0.5)

pipeline = dai.Pipeline()

# --- RGB-Kamera für STILL (Einzelbild) ---
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)

# *** WICHTIG: Sensor auf 12 MP setzen ***
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# *** STILL-STREAM für maximale Qualität ***
cam.setStillSize(4032, 3040)      # ? Volle Sensorauflösung!
cam.setPreviewSize(640, 480)      # Optional für Vorschau

# Auto-Fokus
cam.initialControl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
cam.initialControl.setAutoExposureEnable()

# --- Output für Still ---
still_out = pipeline.create(dai.node.XLinkOut)
still_out.setStreamName("still")
cam.still.link(still_out.input)   # ? still, nicht video!

# --- Vorschau-Output (optional) ---
preview_out = pipeline.create(dai.node.XLinkOut)
preview_out.setStreamName("preview")
cam.preview.link(preview_out.input)

# ===== GERÄT STARTEN =====
print("Starte Kamera mit USB2-Modus...")
with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
    print("? Kamera bereit")
    
    still_queue = device.getOutputQueue("still", maxSize=1, blocking=True)
    preview_queue = device.getOutputQueue("preview", maxSize=1, blocking=True)
    
    # Fokus stabilisieren lassen
    time.sleep(1.0)
    
    print("?? Löse 12 MP Bild aus...")
    cam.initialControl.setCaptureStill(True)
    
    # Bild abholen
    still_frame = still_queue.get().getCvFrame()
    h, w = still_frame.shape[:2]
    print(f"?? Bildauflösung: {w} x {h} Pixel")
    
    # RGB ? BGR
    frame_bgr = cv2.cvtColor(still_frame, cv2.COLOR_RGB2BGR)
    
    # ===== ALS PNG SPEICHERN (VERLUSTFREI) =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"oak_12MP_STILL_{timestamp}.png"
    cv2.imwrite(filename, frame_bgr)
    
    file_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"?? Gespeichert: {filename} ({file_size:.1f} MB)")
    
    # Vorschau anzeigen
    preview_frame = preview_queue.get().getCvFrame()
    preview_bgr = cv2.cvtColor(preview_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Vorschau (640x480)", preview_bgr)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

control_light(False)
print("\n? Test abgeschlossen.\n")
