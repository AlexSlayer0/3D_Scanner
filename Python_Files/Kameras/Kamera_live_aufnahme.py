from pygrabber.dshow_graph import FilterGraph
import cv2

def list_cameras():
    graph = FilterGraph()
    return graph.get_input_devices()

def live_view_all_cameras():
    devices = list_cameras()
    if not devices:
        print("Keine Kameras gefunden!")
        return

    caps = []

    # Alle Kameras öffnen
    for idx, name in enumerate(devices):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Kamera {idx} gestartet: {name}")
            caps.append((idx, cap))
        else:
            print(f"Kamera {idx} konnte nicht geöffnet werden")

    if not caps:
        print("Keine Kamera konnte geöffnet werden")
        return

    print("ESC drücken zum Beenden")

    while True:
        for idx, cap in caps:
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Kamera {idx}", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    # Aufräumen
    for _, cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_view_all_cameras()
