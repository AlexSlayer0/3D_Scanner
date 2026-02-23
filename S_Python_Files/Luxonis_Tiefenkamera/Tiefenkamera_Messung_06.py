#!/usr/bin/env python3
"""
Stabile 3D-Volumenmessung mit OAK-D2S
Optimiert für DepthAI 2.29
- HIGH_ACCURACY + Subpixel
- Multi-Frame Median
- Pipeline nur einmal gestartet
"""

import cv2
import depthai as dai
import numpy as np
import json
import os
import time
from typing import Dict

# ================= CONFIG =================

class Config:
    ROI_X_MIN = -250
    ROI_X_MAX = 250
    ROI_Y_MIN = -250
    ROI_Y_MAX = 250

    MIN_OBJEKT_HOEHE_MM = 5.0
    MAX_OBJEKT_HOEHE_MM = 300.0
    MAX_LAENGE_BREITE_MM = 450.0

    CALIB_FILE = "distanz_calibration.json"
    MULTI_FRAMES = 6  # Anzahl Frames für Median

# ================= KLASSE =================

class OakD2Volume:

    def __init__(self):
        self.pipeline = self._create_pipeline()
        self.device = dai.Device(self.pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER)
        self.q_depth = self.device.getOutputQueue("depth", maxSize=4, blocking=True)
        self.q_pc = self.device.getOutputQueue("pointcloud", maxSize=4, blocking=True)

        self.fx, self.fy, self.cx, self.cy = self._read_intrinsics()
        self.calibration = self._load_calibration()

    # ---------- Pipeline ----------
    def _create_pipeline(self):
        pipeline = dai.Pipeline()

        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)

        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = pipeline.create(dai.node.StereoDepth)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        stereo.setSubpixel(True)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)

        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_3x3)
        stereo.initialConfig.setConfidenceThreshold(200)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        pointcloud = pipeline.create(dai.node.PointCloud)
        stereo.depth.link(pointcloud.inputDepth)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        xout_pc = pipeline.create(dai.node.XLinkOut)
        xout_pc.setStreamName("pointcloud")
        pointcloud.outputPointCloud.link(xout_pc.input)

        return pipeline

    # ---------- Intrinsics ----------
    def _read_intrinsics(self):
        calib = self.device.readCalibration()
        intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B)

        fx = intr[0][0]
        fy = intr[1][1]
        cx = intr[0][2]
        cy = intr[1][2]

        print(f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}")
        return fx, fy, cx, cy

    # ---------- Multi Frame Capture ----------
    def _capture_pointcloud(self):
        frames = []
        pcs = []

        for _ in range(Config.MULTI_FRAMES):
            depth = self.q_depth.get()
            pc = self.q_pc.get()

            frames.append(depth.getCvFrame())
            pcs.append(pc.getPoints() * 1000)  # mm

        depth_med = np.median(np.stack(frames), axis=0).astype(np.uint16)
        pc_med = np.median(np.stack(pcs), axis=0)

        return depth_med, pc_med

    # ---------- Kalibrierung ----------
    def calibrate_empty(self):
        print("Kalibrierung – bitte Box leer lassen...")
        time.sleep(1)

        _, points = self._capture_pointcloud()

        x, y, z = points[:,0], points[:,1], points[:,2]

        mask = (z > 0) & \
               (x > Config.ROI_X_MIN) & (x < Config.ROI_X_MAX) & \
               (y > Config.ROI_Y_MIN) & (y < Config.ROI_Y_MAX)

        z_roi = z[mask]

        data = {
            "z_median": float(np.median(z_roi)),
            "z_std": float(np.std(z_roi))
        }

        with open(Config.CALIB_FILE, "w") as f:
            json.dump(data, f, indent=2)

        print("Kalibrierung gespeichert.")
        self.calibration = data

    def _load_calibration(self):
        if os.path.exists(Config.CALIB_FILE):
            with open(Config.CALIB_FILE, "r") as f:
                return json.load(f)
        return None

    # ---------- Messung ----------
    def measure(self) -> Dict:

        if self.calibration is None:
            return {"success": False, "error": "Keine Kalibrierung"}

        depth_frame, points = self._capture_pointcloud()

        x, y, z = points[:,0], points[:,1], points[:,2]

        mask = (z > 0) & \
               (x > Config.ROI_X_MIN) & (x < Config.ROI_X_MAX) & \
               (y > Config.ROI_Y_MIN) & (y < Config.ROI_Y_MAX)

        x, y, z = x[mask], y[mask], z[mask]

        z_median = self.calibration["z_median"]
        tolerance = 3 * self.calibration["z_std"]

        obj_mask = np.abs(z - z_median) > tolerance

        if np.sum(obj_mask) < 30:
            return {"success": False, "error": "Kein Objekt erkannt"}

        x, y, z = x[obj_mask], y[obj_mask], z[obj_mask]

        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        min_z = np.min(z)

        length = max_x - min_x
        width = max_y - min_y
        height = z_median - min_z

        if length < width:
            length, width = width, length

        volume = length * width * height

        vis = self._visualize(depth_frame)

        return {
            "success": True,
            "length": round(length,1),
            "width": round(width,1),
            "height": round(height,1),
            "volume": round(volume,1),
            "frame": vis
        }

    # ---------- Visualisierung ----------
    def _visualize(self, depth):
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)


# ================= MAIN =================

if __name__ == "__main__":

    oak = OakD2Volume()

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        oak.calibrate_empty()
    else:
        result = oak.measure()

        if result["success"]:
            print("\nMESSUNG:")
            print("Länge:", result["length"], "mm")
            print("Breite:", result["width"], "mm")
            print("Höhe:", result["height"], "mm")
            print("Volumen:", result["volume"], "mm³")

            cv2.imshow("Depth", result["frame"])
            cv2.waitKey(0)
        else:
            print("Fehler:", result["error"])#!/usr/bin/env python3
"""
Stabile 3D-Volumenmessung mit OAK-D2S
Optimiert für DepthAI 2.29
- HIGH_ACCURACY + Subpixel
- Multi-Frame Median
- Pipeline nur einmal gestartet
"""

import cv2
import depthai as dai
import numpy as np
import json
import os
import time
from typing import Dict

# ================= CONFIG =================

class Config:
    ROI_X_MIN = -250
    ROI_X_MAX = 250
    ROI_Y_MIN = -250
    ROI_Y_MAX = 250

    MIN_OBJEKT_HOEHE_MM = 5.0
    MAX_OBJEKT_HOEHE_MM = 300.0
    MAX_LAENGE_BREITE_MM = 450.0

    CALIB_FILE = "distanz_calibration.json"
    MULTI_FRAMES = 6  # Anzahl Frames für Median

# ================= KLASSE =================

class OakD2Volume:

    def __init__(self):
        self.pipeline = self._create_pipeline()
        self.device = dai.Device(self.pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH)
        self.q_depth = self.device.getOutputQueue("depth", maxSize=4, blocking=True)
        self.q_pc = self.device.getOutputQueue("pointcloud", maxSize=4, blocking=True)

        self.fx, self.fy, self.cx, self.cy = self._read_intrinsics()
        self.calibration = self._load_calibration()

    # ---------- Pipeline ----------
    def _create_pipeline(self):
        pipeline = dai.Pipeline()

        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)

        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = pipeline.create(dai.node.StereoDepth)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        stereo.setSubpixel(True)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)

        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_3x3)
        stereo.initialConfig.setConfidenceThreshold(200)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        pointcloud = pipeline.create(dai.node.PointCloud)
        stereo.depth.link(pointcloud.inputDepth)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        xout_pc = pipeline.create(dai.node.XLinkOut)
        xout_pc.setStreamName("pointcloud")
        pointcloud.outputPointCloud.link(xout_pc.input)

        return pipeline

    # ---------- Intrinsics ----------
    def _read_intrinsics(self):
        calib = self.device.readCalibration()
        intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B)

        fx = intr[0][0]
        fy = intr[1][1]
        cx = intr[0][2]
        cy = intr[1][2]

        print(f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}")
        return fx, fy, cx, cy

    # ---------- Multi Frame Capture ----------
    def _capture_pointcloud(self):
        frames = []
        pcs = []

        for _ in range(Config.MULTI_FRAMES):
            depth = self.q_depth.get()
            pc = self.q_pc.get()

            frames.append(depth.getCvFrame())
            pcs.append(pc.getPoints() * 1000)  # mm

        depth_med = np.median(np.stack(frames), axis=0).astype(np.uint16)
        pc_med = np.median(np.stack(pcs), axis=0)

        return depth_med, pc_med

    # ---------- Kalibrierung ----------
    def calibrate_empty(self):
        print("Kalibrierung – bitte Box leer lassen...")
        time.sleep(1)

        _, points = self._capture_pointcloud()

        x, y, z = points[:,0], points[:,1], points[:,2]

        mask = (z > 0) & \
               (x > Config.ROI_X_MIN) & (x < Config.ROI_X_MAX) & \
               (y > Config.ROI_Y_MIN) & (y < Config.ROI_Y_MAX)

        z_roi = z[mask]

        data = {
            "z_median": float(np.median(z_roi)),
            "z_std": float(np.std(z_roi))
        }

        with open(Config.CALIB_FILE, "w") as f:
            json.dump(data, f, indent=2)

        print("Kalibrierung gespeichert.")
        self.calibration = data

    def _load_calibration(self):
        if os.path.exists(Config.CALIB_FILE):
            with open(Config.CALIB_FILE, "r") as f:
                return json.load(f)
        return None

    # ---------- Messung ----------
    def measure(self) -> Dict:

        if self.calibration is None:
            return {"success": False, "error": "Keine Kalibrierung"}

        depth_frame, points = self._capture_pointcloud()

        x, y, z = points[:,0], points[:,1], points[:,2]

        mask = (z > 0) & \
               (x > Config.ROI_X_MIN) & (x < Config.ROI_X_MAX) & \
               (y > Config.ROI_Y_MIN) & (y < Config.ROI_Y_MAX)

        x, y, z = x[mask], y[mask], z[mask]

        z_median = self.calibration["z_median"]
        tolerance = 3 * self.calibration["z_std"]

        obj_mask = np.abs(z - z_median) > tolerance

        if np.sum(obj_mask) < 30:
            return {"success": False, "error": "Kein Objekt erkannt"}

        x, y, z = x[obj_mask], y[obj_mask], z[obj_mask]

        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        min_z = np.min(z)

        length = max_x - min_x
        width = max_y - min_y
        height = z_median - min_z

        if length < width:
            length, width = width, length

        volume = length * width * height

        vis = self._visualize(depth_frame)

        return {
            "success": True,
            "length": round(length,1),
            "width": round(width,1),
            "height": round(height,1),
            "volume": round(volume,1),
            "frame": vis
        }

    # ---------- Visualisierung ----------
    def _visualize(self, depth):
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)


# ================= MAIN =================

if __name__ == "__main__":

    oak = OakD2Volume()

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        oak.calibrate_empty()
    else:
        result = oak.measure()

        if result["success"]:
            print("\nMESSUNG:")
            print("Länge:", result["length"], "mm")
            print("Breite:", result["width"], "mm")
            print("Höhe:", result["height"], "mm")
            print("Volumen:", result["volume"], "mm³")

            cv2.imshow("Depth", result["frame"])
            cv2.waitKey(0)
        else:
            print("Fehler:", result["error"])