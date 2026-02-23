import depthai as dai
import json
import time
import numpy as np
import os

# ===== Konfiguration =====
OUTPUT_FILE = "distanz_messung.json"

# Pipeline aufbauen
pipeline = dai.Pipeline()
monoL = pipeline.create(dai.node.MonoCamera)
monoR = pipeline.create(dai.node.MonoCamera)
monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

monoL.out.link(stereo.left)
monoR.out.link(stereo.right)

depthOut = pipeline.create(dai.node.XLinkOut)
depthOut.setStreamName("depth")
stereo.depth.link(depthOut.input)

# Gerät starten
with dai.Device(pipeline) as device:
    qDepth = device.getOutputQueue("depth", maxSize=1, blocking=True)
    depthFrame = qDepth.get().getCvFrame()  # einzelne Aufnahme

# Punkte aus dem Depth Frame
valid_pixels = depthFrame[depthFrame > 0]
mean_distance_mm = float(np.mean(valid_pixels))

# Messung speichern
timestamp = time.strftime("%Y%m%d_%H%M%S")
data = {
    "timestamp": timestamp,
    "mean_mm": mean_distance_mm,
    "min_mm": int(np.min(valid_pixels)),
    "max_mm": int(np.max(valid_pixels)),
    "pixel_count": int(valid_pixels.size)
}

# JSON abspeichern
if os.path.exists(OUTPUT_FILE):
    # vorhandene Daten laden und anhängen
    with open(OUTPUT_FILE, "r") as f:
        existing = json.load(f)
else:
    existing = []

existing.append(data)

with open(OUTPUT_FILE, "w") as f:
    json.dump(existing, f, indent=2)

print(f"Messung gespeichert: {mean_distance_mm:.1f} mm → {OUTPUT_FILE}")