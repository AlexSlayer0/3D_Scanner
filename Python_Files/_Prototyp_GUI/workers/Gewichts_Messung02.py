from operator import index
import sys
import time
import json
import os

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
            return 500  # beliebiger Rohwert
    adc_channels = [DummyADC() for _ in range(3)]
else:
    adc_channels = [NAU7802() for _ in range(3)]

mux_channels = [0, 1, 2]

# ------------------------------
# Persistenz: JSON-Datei
# ------------------------------
PARAM_FILE = "calibration.json"

# Standardwerte (werden beim 1. Start verwendet)
DEFAULT_FAKTOREN = [-0.004423, -0.00425, -0.004543]
DEFAULT_ZERO_OFFSETS = [0.0, 0.0, 0.0]

# Diese Variablen werden aus der Datei geladen bzw. gespeichert
faktoren = DEFAULT_FAKTOREN.copy()
zero_offsets = DEFAULT_ZERO_OFFSETS.copy()

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
            print("Parametergeladen aus", PARAM_FILE)
        except Exception as e:
            print(f"Fehler beim Laden der Parameter: {e}. Verwende Standardwerte.")
    else:
        print("Keine gespeicherten Parameter gefunden. Verwende Standardwerte.")

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
        print("Parameter gespeichert in", PARAM_FILE)
    except Exception as e:
        print(f"Fehler beim Speichern der Parameter: {e}")

# ------------------------------
# MUX umschalten
# ------------------------------
def select_mux(channel):
    if not IS_TEST:
        bus.write_byte(MUX_ADDR, 1 << channel)
        time.sleep(0.05)

# ------------------------------
# ADC-Initialisierung
# ------------------------------
def init_adc():
    """ADC initialisieren"""
    for index in range(len(adc_channels)):
        select_mux(mux_channels[index])
        if not adc_channels[index].begin():
            raise RuntimeError(f"ADC {index} nicht erreichbar")
        #print(f"ADC {index} initialisiert")

# ------------------------------
# Tara-Funktion (mit automatischem Speichern)
# ------------------------------
def tara():
    """Tara für alle Zellen"""
    for i in range(len(adc_channels)):
        if IS_TEST:
            zero_offsets[i] = 0  # Dummy-Offset
        else:
            select_mux(mux_channels[i])
            zero_offsets[i] = adc_channels[i].getAverage(20)
        print(f"Zelle {i}: Zero-Offset = {zero_offsets[i]:.2f}")
    # Nach jeder Tara werden die neuen Offsets gespeichert
    save_params()

# ------------------------------
# Kalibrierung einer Zelle (mit automatischem Speichern)
# ------------------------------
def calibrate_cell(index, known_weight_grams):
    """Kalibriert eine einzelne Zelle mit einem bekannten Gewicht.
    
    Args:
        index (int): Index der zu kalibrierenden Zelle (0, 1, 2)
        known_weight_grams (float): Bekanntes Gewicht in Gramm, das auf der Zelle liegt
    """    
    # MUX auf gewünschte Zelle schalten
    select_mux(mux_channels[index])
    
    # Rohwert mit Tara-Offset messen
    roh = adc_channels[index].getAverage(20) - zero_offsets[index]
    
    if abs(roh) < 1:  # Vermeide Division durch (fast) Null
        #print(f"Warnung: Rohwert zu nahe an Null ({roh:.2f})")
        return faktoren[index]
    
    # Neuen Kalibrierfaktor berechnen: Gewicht = Rohwert * Faktor
    neuer_faktor = known_weight_grams / roh
    
    # Negatives Vorzeichen für Dehnungsmessstreifen typisch, sicherstellen
    if neuer_faktor > 0:
        neuer_faktor = -neuer_faktor
    
    faktoren[index] = neuer_faktor
    
    #print(f"Zelle {index}: Neuer Faktor = {faktoren[index]:.6f}")
    #print(f"  Rohwert: {roh:.2f}, Referenz: {known_weight_grams}g")
    
    # Nach Kalibrierung speichern
    save_params()
    
    return faktoren[index]

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
    
    #print(f"Zelle {index}: {gewicht:.2f} g")
    return gewicht

# ==============================
# Gewichtsmessung aller Zellen
# ==============================
def get_weight():
    """Gibt das Gesamtgewicht aller Zellen zurück"""
    try:
        gesamt = 0
        for i in range(len(adc_channels)):
            gesamt += measure_cell(i)
            time.sleep(0.1)  # Kurze Pause zwischen den Messungen
        #print(f"Gesamtgewicht: {gesamt:.2f} g\n")
        return gesamt
    except Exception as e:
        #print(f"Fehler bei der Gewichtsmessung: {e}")
        return "Undefiniert"

# ==============================
# Hauptprogramm
# ==============================
if __name__ == "__main__":
    print("Starte Messung der 3 Zellen…\n")
    
    # Gespeicherte Parameter laden
    load_params()
    
    # ADCs initialisieren
    init_adc()

    # Tara durchführen (speichert automatisch)
    tara()

    print("Alle ADCs initialisiert und tarriert!\n")
    
    # Beispiel Kalibrierung (auskommentiert)
    # input("Lege ein Referenzgewicht auf Zelle 0 und drücke Enter zum Kalibrieren…")
    # calibrate_cell(0, 1000.0)  # Kalibriere Zelle 0 mit 1000g

    # Messung starten
    while True:
        get_weight()
        time.sleep(2)
