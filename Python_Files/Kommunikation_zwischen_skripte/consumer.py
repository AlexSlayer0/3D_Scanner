import cv2, numpy as np, socket, pickle
from multiprocessing import shared_memory

# Verbindung herstellen
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 5050))

# Infos empfangen
data = b""
while True:
    packet = sock.recv(4096)
    if not packet: break
    data += packet
sock.close()

info = pickle.loads(data)
shape = tuple(info["shape"])
dtype = np.dtype(info["dtype"])
shm = shared_memory.SharedMemory(name=info["name"])
img = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

# Bild anzeigen
cv2.imshow("RAM-Bild", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
