import subprocess

# Worker starten und direkt nur die Ausgabe abholen
proc = subprocess.Popen(
    ["python", "worker.py"],
    stdout=subprocess.PIPE,
    text=True
)

# Eine einzelne Zeile lesen
line = proc.stdout.readline().strip()
print("Ausgabe vom Worker:", line)
