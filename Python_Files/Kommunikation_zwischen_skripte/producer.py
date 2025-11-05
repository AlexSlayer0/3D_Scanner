import cv2, numpy as np, socket, pickle
from multiprocessing import shared_memory

# Kamera aufnehmen
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Kamera-Fehler")

# Shared Memory anlegen
shm = shared_memory.SharedMemory(create=True, size=frame.nbytes)
np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)[:] = frame[:]

# Infos für den Consumer vorbereiten
info = {
    "name": shm.name,
    "shape": frame.shape,
    "dtype": str(frame.dtype)
}

# Über lokalen Socket senden
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("127.0.0.1", 5050))
sock.listen(1)
print("Warte auf Consumer ...")
conn, _ = sock.accept()
conn.send(pickle.dumps(info))
conn.close()
print("Info gesendet, Shared Memory aktiv. STRG+C zum Beenden.")

try:
    while True:
        pass
except KeyboardInterrupt:
    shm.close()
    shm.unlink()
