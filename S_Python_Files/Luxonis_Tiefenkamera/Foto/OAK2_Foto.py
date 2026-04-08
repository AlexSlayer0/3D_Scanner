#!/usr/bin/env python3
import depthai as dai
import cv2
import time
from datetime import datetime

# Pipeline bauen
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
cam.setStillSize(4032, 3040)
cam.setPreviewSize(640, 480)          # wird für nichts gebraucht, aber Kamera mag es
cam.setInterleaved(False)

# Auto-Fokus
cam.initialControl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.CONTINUOUS_VIDEO)

# Outputs
still_out = pipeline.create(dai.node.XLinkOut)
still_out.setStreamName("still")
cam.still.link(still_out.input)

control_in = pipeline.create(dai.node.XLinkIn)
control_in.setStreamName("control")
control_in.out.link(cam.inputControl)

# Device starten
with dai.Device(pipeline) as device:
    # Queues
    still_queue = device.getOutputQueue("still", maxSize=1, blocking=False)
    control_queue = device.getInputQueue("control")
    
    # Kurz warten (Autofokus)
    time.sleep(2)
    
    # Bild auslösen
    ctrl = dai.CameraControl()
    ctrl.setCaptureStill(True)
    control_queue.send(ctrl)
    
    # Auf Bild warten (max. 5 Sekunden)
    start = time.time()
    frame = None
    while time.time() - start < 5:
        packet = still_queue.tryGet()
        if packet is not None:
            frame = packet.getCvFrame()   # bereits BGR
            break
        time.sleep(0.1)
    
    if frame is None:
        print("Fehler: Kein Bild empfangen")
        exit(1)
    
    # Speichern
    filename = f"oak_12MP_{datetime.now():%Y%m%d_%H%M%S}.png"
    cv2.imwrite(filename, frame)
    print(f"Bild gespeichert: {filename} ({frame.shape[1]}x{frame.shape[0]})")