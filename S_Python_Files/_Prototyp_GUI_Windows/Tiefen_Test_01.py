import depthai as dai
import cv2

# Pipeline
verfügbar = len(dai.Device.getAllAvailableDevices()) > 0
if not verfügbar:
    print("Keine OAK-D2 Kamera gefunden!")

pipeline = dai.Pipeline()

# 1. RGB-Kamera (so wie bei Ihren Mono-Kameras)
rgb_cam = pipeline.create(dai.node.ColorCamera)  # ColorCamera für RGB
rgb_cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)  # Position A ist RGB-Kamera
rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# 2. XLinkOut Node MUSS existieren - für Verbindung zum PC
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")

# 3. Verbinden (wie bei stereo.left/right)
rgb_cam.video.link(xout_rgb.input)  # Video-Stream an XLinkOut senden

# 4. Mit Gerät verbinden
with dai.Device(pipeline) as device:
    # Queue vom Device holen (nicht vom Node!)
    q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
    
    # Einzelnes Bild holen
    frame = q_rgb.get().getCvFrame()
    cv2.imwrite("rgb_bild.jpg", frame)
    print("Bild gespeichert!")