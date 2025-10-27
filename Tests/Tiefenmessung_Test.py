import depthai as dai
import cv2
import numpy as np

# ===== Pipeline =====
pipeline = dai.Pipeline()

# Mono-Kameras für Stereo
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Stereo Depth
stereo = pipeline.create(dai.node.StereoDepth)
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)

# RGB-Kamera
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setPreviewSize(300, 300)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Neural Network
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath("mobilenet-ssd.blob")
cam.preview.link(nn.input)

# Output Queues
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam.preview.link(xout_rgb.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# ===== Gerät starten =====
with dai.Device(pipeline) as device:
    rgb_q = device.getOutputQueue("rgb", 4, False)
    nn_q = device.getOutputQueue("nn", 4, False)
    depth_q = device.getOutputQueue("depth", 4, False)

    while True:
        rgb_frame = rgb_q.get().getCvFrame()
        depth_frame = depth_q.get().getFrame()

        # Inference auslesen
        in_nn = nn_q.tryGet()
        if in_nn:
            # Annahme: NN gibt klassische Mobilenet-SSD-Struktur zurück
            detections = in_nn.getLayerFp16("detection_out")
            if len(detections) > 0:
                for i in range(0, len(detections), 7):
                    image_id, label, conf, x1, y1, x2, y2 = detections[i:i+7]
                    if conf < 0.5:
                        continue

                    x1 = int(x1 * rgb_frame.shape[1])
                    y1 = int(y1 * rgb_frame.shape[0])
                    x2 = int(x2 * rgb_frame.shape[1])
                    y2 = int(y2 * rgb_frame.shape[0])

                    cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    depth_crop = depth_frame[y1:y2, x1:x2]
                    if depth_crop.size == 0:
                        continue

                    mean_depth_m = np.mean(depth_crop) / 1000
                    pixel_width = x2 - x1
                    pixel_height = y2 - y1
                    f = 0.002
                    width_m = pixel_width * mean_depth_m * f
                    height_m = pixel_height * mean_depth_m * f

                    cv2.putText(rgb_frame,
                                f"W:{width_m:.2f}m H:{height_m:.2f}m D:{mean_depth_m:.2f}m",
                                (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("RGB", rgb_frame)
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
