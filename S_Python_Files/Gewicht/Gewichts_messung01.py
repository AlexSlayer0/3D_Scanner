import smbus2
import time
from collections import deque
from PyNAU7802 import NAU7802

# I2C Konfiguration
I2C_BUS = 1       # Raspberry Pi oder ESP32 I2C-Bus
MUX_ADDR = 0x70   # PCA9548A Adresse

bus = smbus2.SMBus(I2C_BUS)
adc_channels = [NAU7802() for _ in range(3)]
mux_channels = [0, 1, 2]


# Historie für Stabilität
history_len = 5
history = deque(maxlen=history_len)

def select_mux(channel):
    """Aktiviere einen bestimmten Kanal des PCA9548A"""
    bus.write_byte(MUX_ADDR, 1 << channel)
    time.sleep(0.01)  # kleines Delay für Stabilität

def kalibriere():
    for i, adc in enumerate(adc_channels):
        select_mux(mux_channels[i])
        adc.calibrate()
    print("Kalibrierung abgeschlossen")

def tara():
    for i, adc in enumerate(adc_channels):
        select_mux(mux_channels[i])
        adc.tare()
    print("Tara abgeschlossen")

def messung():
    werte = []
    for i, adc in enumerate(adc_channels):
        select_mux(mux_channels[i])
        w = adc.get_weight()  # Rohwert vom ADC
        werte.append(w)
    gesamtgewicht = sum(werte)

    # Stabilitätscheck
    history.append(gesamtgewicht)
    if any(abs(gesamtgewicht - h) < 0.01 for h in history):  # ±10g Toleranz
        print(f"Gewicht stabil erkannt: {gesamtgewicht:.2f} kg")
    return gesamtgewicht

# Hauptschleife
while True:
    messung()
    time.sleep(0.2)
