import sys
import time

# ==============================
# Test-Modus für Windows
# ==============================
IS_TEST = sys.platform != "linux"

if not IS_TEST:
    import smbus2
    from PyNAU7802 import NAU7802

# ==============================
# I2C & MUX (nur Linux)
# ==============================
I2C_BUS = 1
MUX_ADDR = 0x70
if not IS_TEST:
    bus = smbus2.SMBus(I2C_BUS)

# Wir gehen von 3 Zellen aus
if IS_TEST:
    class DummyADC:
        def begin(self):
            return True
        def getAverage(self, n):
            # Simuliere einen festen Rohwert für Test
            return 1000  # beliebiger Rohwert
    adc_channels = [DummyADC() for _ in range(3)]
else:
    adc_channels = [NAU7802() for _ in range(3)]

mux_channels = [0, 1, 2]

# ==============================
# Kalibrierfaktoren & Zero-Offsets
# ==============================
faktoren = [-0.004423, -0.00425, -0.004543]
zero_offsets = [0.0] * len(adc_channels)

# ==============================
# MUX umschalten
# ==============================
def select_mux(channel):
    if not IS_TEST:
        bus.write_byte(MUX_ADDR, 1 << channel)
        time.sleep(0.05)

# ==============================
# ADC initialisieren
# ==============================
def init_adc(index):
    select_mux(mux_channels[index])
    if not adc_channels[index].begin():
        raise RuntimeError(f"ADC {index} nicht erreichbar")
    print(f"ADC {index} initialisiert")

# ==============================
# Tara für alle Zellen
# ==============================
def tara():
    print("Leere Zellen auflegen für Tara…")
    for i in range(len(adc_channels)):
        if IS_TEST:
            zero_offsets[i] = 0  # Dummy-Offset
        else:
            input(f"Zelle {i} leer lassen und Enter drücken…")
            select_mux(mux_channels[i])
            zero_offsets[i] = adc_channels[i].getAverage(20)
        print(f"Zelle {i}: Zero-Offset = {zero_offsets[i]:.2f}")
    print("Tara abgeschlossen\n")

# ==============================
# Einzelmessung einer Zelle
# ==============================
gewicht_alt_list = [0.0] * len(adc_channels)

def measure_cell(index):
    global gewicht_alt_list
    select_mux(mux_channels[index])
    
    if IS_TEST:
        roh = 1_000_000  # Dummy-Rohwert für 1 kg
    else:
        roh = adc_channels[index].getAverage(10) - zero_offsets[index]
    
    alpha = 0.5  # Glättungsfaktor
    gewicht = roh * faktoren[index] if not IS_TEST else 1000  # 1 kg fix
    
    # Glätten
    gewicht = alpha * gewicht + (1 - alpha) * gewicht_alt_list[index]
    gewicht_alt_list[index] = gewicht
    
    print(f"Zelle {index}: {gewicht:.2f} g")
    return gewicht

# ==============================
# Hauptprogramm
# ==============================
if __name__ == "__main__":
    print("Starte Messung der 3 Zellen…\n")
    
    # ADCs initialisieren
    for i in range(len(adc_channels)):
        init_adc(i)

    # Tara durchführen
    tara()

    print("Alle ADCs initialisiert und tarriert!\n")
    
    # Messung starten
    while True:
        gesamt = 0
        for i in range(len(adc_channels)):
            gesamt += measure_cell(i)
        print(f"Gesamtgewicht: {gesamt:.2f} g\n")
        time.sleep(0.5)
