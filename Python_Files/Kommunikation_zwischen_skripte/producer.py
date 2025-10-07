import cv2
import numpy as np
from multiprocessing import shared_memory

bilder = []
shm_list = []

for i in range(4):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Bild {i} konnte nicht aufgenommen werden")
    bilder.append(frame)

    # Shared Memory für jedes Bild erstellen
    shm = shared_memory.SharedMemory(create=True, size=frame.nbytes)
    shm_array = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
    shm_array[:] = frame[:]
    shm_list.append(shm)

    print(f"Bild {i} im RAM gespeichert. Shared Memory Name: {shm.name}, Shape: {frame.shape}")

print("Producer fertig. Drücke STRG+C zum Beenden.")
try:
    import time
    print("Producer läuft, SHM bleibt offen. STRG+C zum Beenden.")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    for shm in shm_list:
        shm.close()
        shm.unlink()
    print("Shared Memory freigegeben.")

