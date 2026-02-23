#!/usr/bin/env python3
import depthai as dai
import cv2
import numpy as np
import json
import os

CALIB_FILE = "distanz_messung.json"

# ROI in mm (Ihre neuen Werte)
ROI_X_MIN = -550
ROI_X_MAX = -50
ROI_Y_MIN = -300
ROI_Y_MAX = -50

TOLERANCE = 10       # 10 mm um die Referenzdistanz (nach oben zur Kamera)

def load_reference_distance(calib_file=CALIB_FILE):
    if not os.path.exists(calib_file):
        raise FileNotFoundError(f"Kalibrierungsdatei {calib_file} nicht gefunden")
    with open(calib_file, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = data[0]
    return float(data["mean_mm"])

def main():
    ref_dist = load_reference_distance()
    print(f"Referenzdistanz: {ref_dist:.1f} mm")

    # Pipeline erstellen
    pipeline = dai.Pipeline()
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setRectifyEdgeFillColor(0)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("depth")
    stereo.depth.link(xout.input)

    with dai.Device(pipeline) as device:
        # Intrinsics für die Berechnung von X, Y auslesen
        calib = device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 640, 400)
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]
        cx = intrinsics[0][2]
        cy = intrinsics[1][2]
        print(f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

        q = device.getOutputQueue(name="depth", maxSize=4, blocking=True)

        while True:
            depth_frame = q.get().getCvFrame()          # Tiefe in mm, shape (400, 640)
            h, w = depth_frame.shape

            # Pixelkoordinaten-Gitter
            x_coords = np.arange(w)
            y_coords = np.arange(h)
            xx, yy = np.meshgrid(x_coords, y_coords)

            # Gültige Tiefen und Umrechnung in 3D-Koordinaten
            valid = depth_frame > 0
            Z = depth_frame.astype(np.float32)
            X = (xx - cx) * Z / fx
            Y = (yy - cy) * Z / fy

            print("X[0,0] =", X[0, 0], "Y[0,0] =", Y[0, 0], "Z[0,0] =", Z[0, 0])
            # Maske: ROI in X, Y und Tiefe relativ zur Referenz
            mask = valid
            mask &= (X >= ROI_X_MIN) & (X <= ROI_X_MAX)
            mask &= (Y >= ROI_Y_MIN) & (Y <= ROI_Y_MAX)
            # Objektpunkte sind näher als die Referenz (Z < ref_dist) und innerhalb der Toleranz
            mask &= (Z > ref_dist - TOLERANCE) & (Z < ref_dist)

            print(f"Anzahl gültiger Punkte im ROI: {np.sum(mask)}")

            if np.sum(mask) > 0:
                # Minimalen Z-Wert des Objekts finden
                min_z = np.min(Z[mask])
                hoehe_mm = ref_dist - min_z
                print(f"Objekthöhe: {hoehe_mm:.1f} mm (min Z = {min_z:.1f} mm)")
                # Bounding Box in Pixelkoordinaten (für Visualisierung)
                ys, xs = np.where(mask)
                min_x, max_x = np.min(xs), np.max(xs)
                min_y, max_y = np.min(ys), np.max(ys)

                # Objekt im Tiefenbild markieren
                cv2.rectangle(depth_frame, (min_x, min_y), (max_x, max_y), (0, 255, 255), 2)
                cv2.putText(depth_frame, f"minZ={min_z:.1f} mm", (min_x, min_y-25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(depth_frame, f"Hoehe = {hoehe_mm:.1f} mm", (min_x, min_y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(depth_frame, "Kein Objekt erkannt", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Tiefenbild einfärben und anzeigen
            depth_color = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            depth_color = cv2.applyColorMap(depth_color.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow("Hoehenmessung", depth_color)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()