import smbus2
import time
from PyNAU7802 import NAU7802

# ==============================
# I2C & MUX
# ==============================
I2C_BUS = 1
MUX_ADDR = 0x70
bus = smbus2.SMBus(I2C_BUS)

adc_channels = [NAU7802() for _ in range(1)]
mux_channels = [0]

# ==============================
# Kalibrierung
# ==============================
zero_offsets = [0.0] * len(adc_channels)
faktoren = [0.0] * len(adc_channels)  # später berechnet

# ==============================
# MUX umschalten
# ==============================
def select_mux(channel):
    bus.write_byte(MUX_ADDR, 1 << channel)
    time.sleep(0.05)

# ==============================
# Einzelnen ADC initialisieren
# ==============================
def init_adc(index):
    select_mux(mux_channels[index])
    if not adc_channels[index].begin():
        raise RuntimeError(f"ADC {index} nicht erreichbar")
    print(f"ADC {index} initialisiert")

# ==============================
# Einzelzelle Tara & Kalibrierung
# ==============================
def calibrate_cell(index):
    input(f"Leere Zelle {index} lassen und Enter drücken…")
    select_mux(mux_channels[index])
    zero_offsets[index] = adc_channels[index].getAverage(10)
    print(f"ADC {index}: Zero-Offset = {zero_offsets[index]:.2f}")

    gewicht = float(input(f"Bekanntes Gewicht auf Zelle {index} legen (in g): "))
    select_mux(mux_channels[index])
    raw = adc_channels[index].getAverage(10) - zero_offsets[index]
    faktoren[index] = gewicht / raw
    print(f"ADC {index}: Kalibrierfaktor = {faktoren[index]:.6f}\n")

# ==============================
# Einzelmessung einer Zelle
# ==============================
gewicht_alt_list = [0.0] * len(adc_channels)  # global

def measure_cell(index):
	empty_threshold = 0.3
	global gewicht_alt_list
	select_mux(mux_channels[index])
    
	roh = adc_channels[index].getAverage(10) - zero_offsets[index]
    
	if abs(roh - zero_offsets[0]) < empty_threshold:
		zero_offsets[0] = roh  # Offset aktualisieren
    
	alpha = 0.5  # Glättungsfaktor
	gewicht = roh * faktoren[index]
    
    # Glätten mit vorherigem Wert
	gewicht = alpha * gewicht + (1 - alpha) * gewicht_alt_list[index]
	gewicht_alt_list[index] = gewicht  # speichern für nächsten Durchgang
    
	print(f"Zelle {index}: {gewicht:.2f} g")
	return gewicht

    
# ==============================
# Hauptprogramm für Einzeltest
# ==============================
if __name__ == "__main__":
    print("Starte Einzeltest der Zellen…\n")
    
    # Jeden ADC initialisieren und kalibrieren
    for i in range(len(adc_channels)):
        init_adc(i)
        calibrate_cell(i)

    print("Alle Zellen kalibriert!\n")
    
    print("Starte Messung jeder Zelle…")
    while True:
        gesamt = 0
        for i in range(len(adc_channels)):
            gesamt += measure_cell(i)
        print(f"Gesamtgewicht: {gesamt:.2f} g\n")
        time.sleep(0.5)
