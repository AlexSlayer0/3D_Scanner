import smbus2
import time

from PyNAU7802 import NAU7802

# ==============================
# I2C & MUX
# ==============================
I2C_BUS = 1
MUX_ADDR = 0x70

bus = smbus2.SMBus(I2C_BUS)
adc_channels = [NAU7802() for _ in range(3)]
mux_channels = [0, 1, 2]

REFERENZ_GEWICHT = 1000.0  # g
zero_offsets = [0.0, 0.0, 0.0]
gesamt_faktor = 1.0

# ==============================
# MUX
# ==============================
def select_mux(channel):
    bus.write_byte(MUX_ADDR, 1 << channel)
    time.sleep(0.02)

# ==============================
# ADC Init
# ==============================
def init_adcs():
    for i, adc in enumerate(adc_channels):
        select_mux(mux_channels[i])
        if not adc.begin():
            raise RuntimeError(f"ADC {i} nicht erreichbar")
    print("✅ ADCs initialisiert\n")

# ==============================
# TARA
# ==============================
def tara():
    input("Waage leer lassen → ENTER")
    for i, adc in enumerate(adc_channels):
        select_mux(mux_channels[i])
        zero_offsets[i] = adc.getAverage(20)
    print("✅ Tara abgeschlossen\n")

# ==============================
# FLÄCHEN-KALIBRIERUNG
# ==============================
def flaechen_kalibrierung():
    print("🔧 Flächen-Kalibrierung startet")
    print("Gewicht nacheinander an jede Position legen\n")

    roh_summen = []

    positionen = [
        "Ecke links oben", "Mitte oben", "Ecke rechts oben",
        "Mitte links", "MITTE", "Mitte rechts",
        "Ecke links unten", "Mitte unten", "Ecke rechts unten"
    ]

    for pos in positionen:
        input(f"{REFERENZ_GEWICHT} g auf {pos} → ENTER")

        roh_gesamt = 0
        for i, adc in enumerate(adc_channels):
            select_mux(mux_channels[i])
            roh = adc.getAverage(20) - zero_offsets[i]
            roh_gesamt += roh

        roh_summen.append(roh_gesamt)
        print(f"  Roh-Summe: {roh_gesamt:.2f}")

    global gesamt_faktor
    roh_mittel = sum(roh_summen) / len(roh_summen)
    gesamt_faktor = REFERENZ_GEWICHT / roh_mittel

    print("\n✅ Kalibrierung abgeschlossen")
    print(f"Gesamtfaktor: {gesamt_faktor:.6f}\n")

# ==============================
# MESSUNG
# ==============================
def messung():
    roh_gesamt = 0
    for i, adc in enumerate(adc_channels):
        select_mux(mux_channels[i])
        roh_gesamt += adc.getAverage(5) - zero_offsets[i]

    gewicht = roh_gesamt * gesamt_faktor
    print(f"Gesamtgewicht: {gewicht:.2f} g")
    return gewicht

# ==============================
# HAUPTPROGRAMM
# ==============================
if __name__ == "__main__":
    init_adcs()
    tara()
    flaechen_kalibrierung()

    print("▶ Messbetrieb\n")
    while True:
        messung()
        time.sleep(0.2)
