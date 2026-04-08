

import depthai as dai
import time
import sys

def print_header(text):
    print("\n" + "="*50)
    print(f" {text}")
    print("="*50)

def check_device():
    """Prüft, ob ein Gerät verfügbar ist und gibt Info zurück"""
    (found, info) = dai.DeviceBootloader.getFirstAvailableDevice()
    if not found:
        print("? KEIN GERÄT GEFUNDEN!")
        print("   Bitte OAK-D2 abziehen, 5 Sekunden warten, wieder einstecken.")
        return None
    return info

def get_device_info():
    info = check_device()
    if info is None:
        return False
    
    print_header("GERÄT INFORMATIONEN")
    print(f"? Gerät gefunden: {info.name}")
    print(f"   MXID: {info.mxid}")
    print(f"   Zustand: {info.state}")
    
    # Bootloader öffnen
    bl = dai.DeviceBootloader(info)
    version = bl.getVersion()
    print(f"\n?? Bootloader Version: {version}")
    
    # Aktuelle Anwendung/Firmware
    try:
        app_info = bl.getApplicationInfo()
        print(f"?? Firmware: {app_info.name}")
        print(f"   Version: {app_info.version}")
        print(f"   Build-Datum: {app_info.buildDate}")
        return bl, info
    except:
        print("??  Keine Anwendung geflasht (leeres Gerät)")
        return bl, info

def flash_latest_firmware(bl, info):
    """Aktualisiert die Firmware auf die neueste Version"""
    print_header("FIRMWARE UPDATE")
    print("?? Suche nach aktueller Firmware...")
    
    try:
        # Verfügbare Firmwares abrufen
        available = bl.getAvailableFirmwares()
        print(f"\n   Verfügbare Versionen:")
        for fw in available:
            print(f"   - {fw.version} ({fw.name})")
        
        # Neueste Version auswählen
        latest = available[0] if available else None
        if latest:
            print(f"\n?? Installiere {latest.version}...")
            bl.flash(bl.FlashType.APPLICATION, latest)
            print("? Firmware erfolgreich aktualisiert!")
            print("   Bitte OAK-D2 neu starten (abziehen/einstecken).")
            return True
        else:
            print("??  Keine neue Firmware verfügbar.")
            return False
            
    except Exception as e:
        print(f"? Firmware-Update fehlgeschlagen: {e}")
        return False

def test_camera_resolutions():
    """Testet, welche Auflösungen die Kamera liefern kann"""
    print_header("KAMERA-TEST")
    
    # Test-Konfigurationen
    test_configs = [
        ("4K", dai.ColorCameraProperties.SensorResolution.THE_4_K, (3840, 2160)),
        ("12 MP", dai.ColorCameraProperties.SensorResolution.THE_12_MP, (4032, 3040)),
    ]
    
    for name, res, expected_size in test_configs:
        print(f"\n?? Teste {name}...")
        try:
            pipeline = dai.Pipeline()
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            cam.setResolution(res)
            cam.setStillSize(*expected_size)
            
            xout = pipeline.create(dai.node.XLinkOut)
            xout.setStreamName("test")
            cam.still.link(xout.input)
            
            with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
                q = device.getOutputQueue("test", maxSize=1, blocking=True)
                cam.initialControl.setCaptureStill(True)
                
                # Timeout nach 3 Sekunden
                import threading
                result = [None]
                
                def get_frame():
                    try:
                        result[0] = q.get().getCvFrame()
                    except:
                        pass
                
                thread = threading.Thread(target=get_frame)
                thread.daemon = True
                thread.start()
                thread.join(timeout=3.0)
                
                if result[0] is not None:
                    h, w = result[0].shape[:2]
                    print(f"   ? Erfolg: {w} x {h}")
                    return True
                else:
                    print(f"   ? Kein Bild erhalten")
                    
        except Exception as e:
            print(f"   ? Fehler: {e}")
    
    return False


def main():
    print_header("DEPTHAI FIRMWARE DIAGNOSE")
    print("Bevor wir beginnen: ZIEHEN SIE DAS OAK-D2 KABEL AB!")
    input("Nach 5 Sekunden wieder einstecken, dann ENTER drücken...")
    
    # 1. Gerät finden
    info = check_device()
    if info is None:
        return
    
    # 2. Aktuelle Firmware auslesen
    bl, info = get_device_info()
    
    # 3. Firmware-Update anbieten
    print_header("FIRMWARE UPDATE")
    choice = input("Firmware aktualisieren? (j/n): ")
    if choice.lower() == 'j':
        flash_latest_firmware(bl, info)
        print("\n??  Bitte OAK-D2 neu starten und Skript erneut ausführen!")
        return
    
    # 4. Kamera testen (nur wenn kein Update durchgeführt)
    test_camera_resolutions()

if __name__ == "__main__":
    main()

