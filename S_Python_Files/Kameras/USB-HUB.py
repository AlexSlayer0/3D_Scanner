'''
from pygrabber.dshow_graph import FilterGraph

def list_cameras():
    graph = FilterGraph()
    devices = graph.get_input_devices()
    if not devices:
        print("⚠️ Keine Kameras gefunden!")
    else:
        print("🎥 Verfügbare Kameras:")
        for i, name in enumerate(devices):
            print(f"[{i}] {name}")

if __name__ == "__main__":
    list_cameras()

'''

from pygrabber.dshow_graph import FilterGraph
import cv2
import os

def list_cameras():
    graph = FilterGraph()
    devices = graph.get_input_devices()
    return devices

def capture_from_cameras(output_dir="camera_photos"):
    devices = list_cameras()
    if not devices:
        print("⚠️ Keine Kameras gefunden!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, name in enumerate(devices):
        print(f"📸 Kamera {idx}: {name} wird aufgenommen...")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Windows: CAP_DSHOW für stabilen Zugriff
        
        if not cap.isOpened():
            print(f"⚠️ Kamera {idx} konnte nicht geöffnet werden!")
            continue
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            filename = os.path.join(output_dir, f"camera_{idx}.png")
            cv2.imwrite(filename, frame)
            print(f"✅ Bild gespeichert: {filename}")
        else:
            print(f"⚠️ Kein Bild von Kamera {idx} erhalten!")

if __name__ == "__main__":
    capture_from_cameras()

