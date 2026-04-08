#!/usr/bin/env python3
import depthai as dai
import cv2
import numpy as np
import math

# -------- Kamera FOV (OAK-D2S RGB)
FOV_HORIZONTAL = 69  # Grad (RGB Kamera)

# -------- Pipeline
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setPreviewSize(640, 400)
cam.setInterleaved(False)

stereo = pipeline.createStereoDepth()
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

monoL = pipeline.createMonoCamera()
monoR = pipeline.createMonoCamera()

monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)

monoL.out.link(stereo.left)
monoR.out.link(stereo.right)

xoutRgb = pipeline.createXLinkOut()
xoutDepth = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")
xoutDepth.setStreamName("depth")

cam.preview.link(xoutRgb.input)
stereo.depth.link(xoutDepth.input)

# -------- Start
with dai.Device(pipeline) as device:

    qRgb = device.getOutputQueue("rgb")
    qDepth = device.getOutputQueue("depth")

    while True:
        inRgb = qRgb.get()
        inDepth = qDepth.get()

        frame = inRgb.getCvFrame()
        depth = inDepth.getFrame()

        h, w = depth.shape

        # ---- ROI Mitte
        roi_size = 80
        cx, cy = w//2, h//2
        roi = depth[cy-roi_size:cy+roi_size,
                    cx-roi_size:cx+roi_size]

        valid = roi[roi > 0]

        if len(valid) > 0:
            distance_mm = np.mean(valid)

            # ---- mm pro Pixel berechnen
            real_width_mm = 2 * distance_mm * math.tan(math.radians(FOV_HORIZONTAL/2))
            mm_per_pixel = real_width_mm / w

            # ---- einfache Konturerkennung
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                c = max(contours, key=cv2.contourArea)
                x,y,ww,hh = cv2.boundingRect(c)

                width_mm  = ww * mm_per_pixel
                height_mm = hh * mm_per_pixel

                cv2.rectangle(frame,(x,y),(x+ww,y+hh),(0,255,0),2)
                cv2.putText(frame,f"{width_mm:.1f} mm",
                            (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(0,255,0),2)

            cv2.putText(frame,f"Distanz: {distance_mm:.1f} mm",
                        (20,40),cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,0,255),2)

        cv2.imshow("Messung", frame)

        if cv2.waitKey(1) == ord('q'):
            break