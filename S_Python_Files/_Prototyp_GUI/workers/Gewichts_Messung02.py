# Gewichts_Messung02.py

import sys
import time
import json
import os
import logging

# Logging setup
logger = logging.getLogger(__name__)

I2C_BUS = 1                 # I2C & MUX (nur Linux)
MUX_ADDR = 0x70             # MUX-Adresse (typisch 0x70 für TCA9548A)
mux_channels = [0, 1, 2]    # MUX-Kanäle für die 3 Zellen

# Persistenz: JSON-Datei
PARAM_FILE = "weight_calibration.json"

# Standardwerte (können durch gespeicherte Werte überschrieben werden)
DEFAULT_FAKTOREN = [-0.004423, -0.00425, -0.004543]
DEFAULT_ZERO_OFFSETS = [0.0, 0.0, 0.0]

faktoren = DEFAULT_FAKTOREN.copy()
zero_offsets = DEFAULT_ZERO_OFFSETS.copy()


# Test-Modus für Windows
IS_TEST = sys.platform != "linux"

if IS_TEST:
    class DummyADC:
        def begin(self):
            return True
        def getAverage(self, n):
            # Simuliere die Libary-Funktion, die den Durchschnitt von n Messungen zurückgibt von PyNAU7802
            return 500  # beliebiger Rohwert
    adc_channels = [DummyADC() for _ in range(3)]
else:
    import smbus2
    from PyNAU7802 import NAU7802
    bus = smbus2.SMBus(I2C_BUS)

    adc_channels = [NAU7802() for _ in range(3)]

# Liste für vorherige Gewichte (für Glättung)
gewicht_alt_list = [0.0] * len(adc_channels)


def load_params():
    """Lädt gespeicherte Faktoren und Offsets aus JSON-Datei.
       Falls Datei nicht existiert, bleiben die Standardwerte erhalten."""
    global faktoren, zero_offsets
    if os.path.exists(PARAM_FILE):
        try:
            with open(PARAM_FILE, 'r') as f:
                data = json.load(f)
            faktoren = data.get('faktoren', DEFAULT_FAKTOREN)
            zero_offsets = data.get('zero_offsets', DEFAULT_ZERO_OFFSETS)
            logger.info(f"Parametergeladen aus {PARAM_FILE}")
        except Exception as e:
            logger.info(f"Fehler beim Laden der Parameter: {e}. Verwende Standardwerte.")
    else:
        logger.warning("Keine gespeicherten Parameter gefunden. Verwende Standardwerte.")


def save_params():
    """Speichert aktuelle Faktoren und Offsets in JSON-Datei.
       Zusätzlich wird ein Kommentar mit den Standard-Faktoren hinterlegt."""
    data = {
        'faktoren': faktoren,
        'zero_offsets': zero_offsets,
        '_comment_default_factors': (
            f"Original calibration factors from code: {DEFAULT_FAKTOREN}. "
            "Copy these values into 'faktoren' to reset manually."
        )
    }
    try:
        with open(PARAM_FILE, 'w') as f:
            json.dump(data, f, indent=4)
            logger.info(f"Parameter gespeichert in {PARAM_FILE}")
    except Exception as e:
        logger.warning(f"Fehler beim Speichern der Parameter: {e}")


def select_mux(channel):
    """Wählt den MUX-Kanal aus, um die entsprechende Zelle zu messen"""
    if not IS_TEST:
        bus.write_byte(MUX_ADDR, 1 << channel)
        time.sleep(0.05)


def init_adc():
    """ADC initialisieren"""
    for index in range(len(adc_channels)):
        select_mux(mux_channels[index])
        if not adc_channels[index].begin():
            raise RuntimeError(f"ADC {index} nicht erreichbar")
        #print(f"ADC {index} initialisiert")


def tara():
    """Tara für alle Zellen"""
    for i in range(len(adc_channels)):
        if IS_TEST:
            zero_offsets[i] = 0  # Dummy-Offset
        else:
            select_mux(mux_channels[i])
            zero_offsets[i] = adc_channels[i].getAverage(20)
        logger.info(f"Zelle {i}: Zero-Offset = {zero_offsets[i]:.2f}")
    # Nach Tara-Offsets speichern
    save_params()


def calibrate_cell(index, known_weight_grams):
    """Kalibriert eine einzelne Zelle mit bekanntem Gewicht (ohne Platte).
    
    Args:
        index (int): Index der Zelle (0,1,2)
        known_weight_grams (float): aufgelegtes Gewicht in Gramm
    """
    select_mux(mux_channels[index])

    # 1. Rohwert ohne Last (Zelle unbelastet)
    raw_zero = adc_channels[index].getAverage(20)
    logger.info(f"Zelle {index}: Rohwert ohne Last = {raw_zero:.2f}")

    # 2. Gewicht auflegen und Rohwert mit Last messen
    input(f"Lege jetzt {known_weight_grams} g auf Zelle {index} und drücke Enter…")
    raw_load = adc_channels[index].getAverage(20)
    logger.info(f"Zelle {index}: Rohwert mit Last = {raw_load:.2f}")

    # 3. Differenz und neuer Faktor
    diff = raw_load - raw_zero
    if abs(diff) < 1:
        logger.info("Fehler: Differenz zu klein - Kalibrierung abgebrochen.")
        return faktoren[index]

    neuer_faktor = known_weight_grams / diff   # positives Vorzeichen
    faktoren[index] = neuer_faktor

    logger.info(f"Zelle {index}: Neuer Faktor = {faktoren[index]:.6f}")
    save_params()
    return faktoren[index]


def measure_cell(index):
    """Misst das Gewicht einer einzelnen Zelle, glättet den Wert und gibt ihn zurück"""
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
    
    #print(f"Zelle {index}: {gewicht:.2f} g")
    return gewicht


def get_weight():
    """Gibt das Gesamtgewicht aller Zellen zurück"""
    try:
        gesamt = 0
        for i in range(len(adc_channels)):
            gesamt += measure_cell(i)
            time.sleep(0.1)  # Kurze Pause zwischen den Messungen
        #logger.info(f"Gesamtgewicht: {gesamt:.2f} g\n")
        return gesamt
    except Exception as e:
        logger.error(f"Fehler bei der Gewichtsmessung: {e}")
        return "Undefiniert"


if __name__ == "__main__":
    """"Hauptprogramm: Initialisiert die ADCs, führt Tara durch und startet die Gewichtsmessung.
       Optional kann eine Kalibrierung mit einem bekannten Gewicht durchgeführt werden."""
    print("Starte Messung der 3 Zellen…\n")
    
    load_params()   # Gespeicherte Parameter laden
    init_adc()      # ADCs initialisieren
    tara()          # Tara durchführen (speichert automatisch)
    print("Alle ADCs initialisiert und tarriert!\n")
    
    # Beispiel Kalibrierung (auskommentiert)
    # input("Lege ein Referenzgewicht auf Zelle 0 und drücke Enter zum Kalibrieren…")
    # calibrate_cell(0, 1000.0)  # Kalibriere Zelle 0 mit 1000g

    # Messung starten
    while True:
        aktuelles_gewicht = get_weight()
        print(f"Aktuelles Gewicht: {aktuelles_gewicht} g")
        time.sleep(2)
