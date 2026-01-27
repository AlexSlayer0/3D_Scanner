# ###########################
# ROI FIXIEREN IN DER MITTE
# Dieses ROI-Fenster ist jetzt immer mittig im Bild.
# ###########################
import cv2
import depthai as dai
import numpy as np

color = (0, 0, 0)

# Berechne die Mitte des Bildes
center_x, center_y = 0.5, 0.5  # Normierte Koordinaten (0 bis 1)
roi_width, roi_height = 0.4, 0.4  # Größe des ROI-Fensters
topLeft = dai.Point2f(center_x - roi_width/2, center_y - roi_height/2)
bottomRight = dai.Point2f(center_x + roi_width/2, center_y + roi_height/2)

# Define sources, outputs and Create pipeline
pipeline = dai.Pipeline()
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

# Linking
monoLeftOut = monoLeft.requestOutput((640, 400))
monoRightOut = monoRight.requestOutput((640, 400))
monoLeftOut.link(stereo.left)
monoRightOut.link(stereo.right)

stereo.setRectification(True)
stereo.setExtendedDisparity(True)

# ROI-Konfiguration
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 10
config.depthThresholds.upperThreshold = 10000
config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.inputConfig.setWaitForMessage(False)
spatialLocationCalculator.initialConfig.addROI(config)

xoutSpatialQueue = spatialLocationCalculator.out.createOutputQueue()
outputDepthQueue = spatialLocationCalculator.passthroughDepth.createOutputQueue()
stereo.depth.link(spatialLocationCalculator.inputDepth)
inputConfigQueue = spatialLocationCalculator.inputConfig.createInputQueue()

mode = 2  # Standardmodus

with pipeline:
    pipeline.start()
    while pipeline.isRunning():
        spatialData = xoutSpatialQueue.get().getSpatialLocations()
        outputDepthImage : dai.ImgFrame = outputDepthQueue.get()
        frameDepth = outputDepthImage.getCvFrame()
        frameDepth = outputDepthImage.getFrame()

        depthFrameColor = cv2.normalize(frameDepth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)

        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])

            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            if mode == 1:
                z_value = int(depthData.spatialCoordinates.z)
            else:
                roi_depth = frameDepth[ymin:ymax, xmin:xmax]
                valid_depth = roi_depth[roi_depth > 0]
                if len(valid_depth) > 0:
                    z_value = int(np.min(valid_depth))
                else:
                    z_value = 0

            text = f"Z: {z_value / 10:.1f} cm" # mm zu cm
            pos = (xmin + 10, ymin - 10)
            scale = 0.5
            thickness = 1
            text_color = (255, 255, 255)  # weiß

            cv2.putText(depthFrameColor, text, pos, fontType, scale, (0,0,0), thickness+2, cv2.LINE_AA)
            cv2.putText(depthFrameColor, text, pos, fontType, scale, text_color, thickness, cv2.LINE_AA)

            # Kalibrierung
            FAKTOR = 1.00  # Beispielwert
            OFFSET = 0     # optional

            z_measured = np.min(z_value)
            print(f"Gemessene Entfernung: {z_measured} mm")

            z_real = z_measured * FAKTOR + OFFSET
            print(f"Kalibrierte Entfernung: {z_real:.1f} mm")

        # Show the frame
        cv2.imshow("depth", depthFrameColor)

        key = cv2.waitKey(1)
        if key == ord('q'):
            pipeline.stop()
            break
        elif key == ord('1'):
            mode = 1
            print("Wechsel zu Standard (Median) Modus")
        elif key == ord('2'):
            mode = 2
            print("Wechsel zu Minimalpunkt Modus")
