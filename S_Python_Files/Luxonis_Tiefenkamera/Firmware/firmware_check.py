#!/usr/bin/env python3
import depthai as dai

(res, info) = dai.DeviceBootloader.getFirstAvailableDevice()

if res:
    print(f"? Gerät gefunden: {info.name}")
    bl = dai.DeviceBootloader(info)
    
    # Bootloader-Version
    bl_version = bl.getVersion()
    print(f"?? Bootloader Version: {bl_version}")
    
    with dai.Device() as device:
        print(f"?? Device Name: {device.getDeviceName()}")
        print(f"?? Connected Cameras: {device.getConnectedCameras()}")
        
        print("\n?? Verfügbare RGB-Auflösungen:")
        rgb_socket = dai.CameraBoardSocket.CAM_A
        if rgb_socket in device.getConnectedCameras():
            sensor = device.getSensorInfo(rgb_socket)
            resolutions = sensor.resolutions
            for res in resolutions:
                if res.width >= 3840:
                    print(f"   - {res.width} x {res.height} ({res.name})")
else:
    print("? Kein Gerät gefunden")
