#!/usr/bin/env python3
import depthai as dai
import cv2

pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")
cam.preview.link(xout.input)

print("Teste OAK-D2 mit USB2-Mode...")
with dai.Device(pipeline, usb2Mode=True) as device:
    q = device.getOutputQueue("rgb", maxSize=1, blocking=True)
    for i in range(5):
        frame = q.get().getCvFrame()
        cv2.imshow("OAK-D2", frame)
        cv2.waitKey(500)
    cv2.destroyAllWindows()
    print("? KAMERA LÄUFT STABIL")
