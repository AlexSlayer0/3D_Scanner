#!/usr/bin/env python3
"""
OAK-D S2 (RVC2) 12MP Still Image Capture
Speziell angepasst für Myriad X VPU mit mehreren Fallback-Optionen
"""

import depthai as dai
import cv2
import time
import serial
import os
import sys
import queue
from datetime import datetime
from pathlib import Path

# ===== PLATTFORM-ERKENNUNG FÜR SERIELLEN PORT =====
def get_serial_port():
    """Ermittelt den richtigen seriellen Port je nach Betriebssystem"""
    if sys.platform.startswith('win'):
        # Windows: Probiere häufige COM-Ports
        for port_num in range(3, 10):
            port = f"COM{port_num}"
            if os.path.exists(f"\\\\.\\{port}"):
                return port
        return "COM3"  # Fallback
    else:
        # Linux / Mac
        return "/dev/ttyUSB0"

# ===== SERIELLE LED-STEUERUNG (OPTIONAL) =====
def control_light(state: bool):
    """Steuert das Licht über serielle Schnittstelle"""
    port = get_serial_port()
    baudrate = 9600
    
    # Überspringe, wenn kein Port angegeben oder im Simulator
    if port is None:
        print("⚠️ Kein serieller Port konfiguriert")
        return
        
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            time.sleep(1.5)
            ser.write(b"Change\n")
            time.sleep(0.1)
            ser.write(b"a\n" if state else b"0\n")
            time.sleep(0.2)
        print(f"Licht {'EIN' if state else 'AUS'}")
    except Exception as e:
        print(f"⚠️ Lichtsteuerung nicht verfügbar: {e}")
        print(f"   Gesuchter Port: {port}")

# ===== PIPELINE FÜR 12MP STILL (RVC2-OPTIMIERT) =====
def create_still_pipeline(use_control=True):
    """
    Erstellt eine DepthAI-Pipeline für 12MP Still-Aufnahmen
    Speziell optimiert für RVC2 (Myriad X)
    """
    pipeline = dai.Pipeline()
    
    # RGB-Kamera konfigurieren
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    
    # Wichtig: Sensor auf 12MP setzen (IMX378) [citation:1]
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam.setStillSize(4032, 3040)      # Volle Sensorauflösung
    cam.setPreviewSize(640, 480)      # Für Vorschau
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    
    # Auto-Fokus für scharfe Bilder
    cam.initialControl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
    cam.initialControl.setAutoExposureEnable()
    
    # Still-Output
    still_out = pipeline.create(dai.node.XLinkOut)
    still_out.setStreamName("still")
    cam.still.link(still_out.input)
    
    # Preview-Output (für Kontrolle)
    preview_out = pipeline.create(dai.node.XLinkOut)
    preview_out.setStreamName("preview")
    cam.preview.link(preview_out.input)
    
    # Control-Input (optional - für RVC2 manchmal nötig)
    if use_control:
        control_in = pipeline.create(dai.node.XLinkIn)
        control_in.setStreamName("control")
        control_in.out.link(cam.inputControl)
    
    return pipeline

# ===== METHODE 1: STILL MIT CONTROL (OFFIZIELLER WEG) =====
def capture_still_with_control(device):
    """Capture mit Control-Queue (empfohlen für RVC2) [citation:2]"""
    print("\n📸 Methode 1: Still mit Control-Queue...")
    
    # Queues einrichten
    still_queue = device.getOutputQueue("still", maxSize=1, blocking=False)
    preview_queue = device.getOutputQueue("preview", maxSize=1, blocking=False)
    control_queue = device.getInputQueue("control")
    
    # Kamera stabilisieren lassen
    time.sleep(2.0)
    
    # Still auslösen
    ctrl = dai.CameraControl()
    ctrl.setCaptureStill(True)
    control_queue.send(ctrl)
    print("   Befehl gesendet, warte auf Bild...")
    
    # Auf Bild warten (mit Timeout)
    start_time = time.time()
    timeout = 10  # 10 Sekunden Maximum
    
    while time.time() - start_time < timeout:
        still_packet = still_queue.tryGet()
        if still_packet is not None:
            frame = still_packet.getCvFrame()
            print(f"   ✅ Bild empfangen nach {time.time()-start_time:.1f}s")
            return frame
        
        # Kleiner Preview als Lebenszeichen
        preview_packet = preview_queue.tryGet()
        if preview_packet is not None:
            print("   📺 Preview aktiv (Kamera läuft)")
        
        time.sleep(0.1)
    
    print("   ⚠️ Timeout - kein Still-Bild empfangen")
    return None

# ===== METHODE 2: VIDEO-STREAM FALLBACK =====
def capture_still_as_video(device):
    """Alternative: Bild aus Video-Stream bei voller Auflösung"""
    print("\n📸 Methode 2: Video-Stream als Fallback...")
    
    # Neue Pipeline mit Video-Stream
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam.setVideoSize(4032, 3040)      # Video in voller Auflösung
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    
    # Outputs
    video_out = pipeline.create(dai.node.XLinkOut)
    video_out.setStreamName("video")
    cam.video.link(video_out.input)
    
    preview_out = pipeline.create(dai.node.XLinkOut)
    preview_out.setStreamName("preview")
    cam.preview.link(preview_out.input)
    
    # Device mit neuer Pipeline starten
    print("   Starte Video-Pipeline...")
    with dai.Device(pipeline) as video_device:
        video_queue = video_device.getOutputQueue("video", maxSize=1, blocking=False)
        preview_queue = video_device.getOutputQueue("preview", maxSize=1, blocking=False)
        
        print("   Warte auf erstes Videobild...")
        start_time = time.time()
        timeout = 5
        
        while time.time() - start_time < timeout:
            video_packet = video_queue.tryGet()
            if video_packet is not None:
                frame = video_packet.getCvFrame()
                print(f"   ✅ Bild empfangen nach {time.time()-start_time:.1f}s")
                return frame
            time.sleep(0.1)
        
        print("   ⚠️ Timeout im Video-Modus")
        return None

# ===== METHODE 3: MINIMAL-PIPELINE (LETZTER VERSUCH) =====
def capture_still_minimal():
    """Einfachste Pipeline ohne Control - nur Still"""
    print("\n📸 Methode 3: Minimal-Pipeline (ohne Control)...")
    
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam.setStillSize(4032, 3040)
    
    # Nur Still-Output
    still_out = pipeline.create(dai.node.XLinkOut)
    still_out.setStreamName("still")
    cam.still.link(still_out.input)
    
    # Automatischer Trigger?
    cam.initialControl.setCaptureStill(True)
    
    print("   Starte Minimal-Pipeline...")
    with dai.Device(pipeline) as device:
        still_queue = device.getOutputQueue("still", maxSize=1, blocking=False)
        
        print("   Warte auf automatisches Still...")
        start_time = time.time()
        timeout = 5
        
        while time.time() - start_time < timeout:
            still_packet = still_queue.tryGet()
            if still_packet is not None:
                frame = still_packet.getCvFrame()
                print(f"   ✅ Bild empfangen nach {time.time()-start_time:.1f}s")
                return frame
            time.sleep(0.1)
        
        print("   ⚠️ Timeout in Minimal-Pipeline")
        return None

# ===== HAUPTFUNKTION =====
def main():
    print("=" * 60)
    print("OAK-D S2 (RVC2) 12MP Still Image Capture")
    print("=" * 60)
    
    # Licht einschalten (optional)
    control_light(True)
    
    # 1. Versuch: Still mit Control-Queue
    print("\n🚀 Starte primäre Pipeline...")
    pipeline = create_still_pipeline(use_control=True)
    
    try:
        with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
            print(f"✅ Gerät verbunden: {device.getDeviceName()}")
            print(f"   USB Geschwindigkeit: {device.getUsbSpeed()}")
            
            # Versuch 1: Offizielle Methode
            frame = capture_still_with_control(device)
            
            if frame is not None:
                process_and_save_image(frame)
            else:
                print("\n🔄 Fallback zu Methode 2...")
                # Device wird automatisch geschlossen
    except Exception as e:
        print(f"⚠️ Fehler in primärer Pipeline: {e}")
    
    # 2. Versuch: Video-Fallback (nur wenn Methode 1 fehlschlug)
    if 'frame' not in locals() or frame is None:
        try:
            frame = capture_still_as_video(None)  # Startet eigene Pipeline
            if frame is not None:
                process_and_save_image(frame)
            else:
                print("\n🔄 Fallback zu Methode 3...")
                frame = capture_still_minimal()
                if frame is not None:
                    process_and_save_image(frame)
                else:
                    print("\n❌ Alle Methoden fehlgeschlagen!")
        except Exception as e:
            print(f"⚠️ Fehler in Fallback-Methoden: {e}")
    
    # Licht ausschalten
    control_light(False)
    print("\n✨ Programm beendet")

def process_and_save_image(frame):
    """Bild speichern – frame ist bereits BGR (von getCvFrame)"""
    h, w = frame.shape[:2]
    print(f"\n📐 Bildauflösung: {w} x {h} Pixel")
    
    # Keine Konvertierung mehr – frame ist schon BGR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"oak_12MP_STILL_{timestamp}.png"
    
    # PNG speichern (verlustfrei)
    cv2.imwrite(filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    
    file_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"💾 Gespeichert: {filename} ({file_size:.1f} MB)")
    
    # Vorschau anzeigen
    preview_small = cv2.resize(frame, (640, 480))
    cv2.imshow("12MP Aufnahme", preview_small)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()