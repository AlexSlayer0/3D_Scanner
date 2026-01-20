/*
Firmware LOLIN D1 Mini
*/

#include <Arduino.h>
#include <FastLED.h>
#include <string>

// ---------- Konfiguration ----------
//RGB Ringe
#define NUM_LEDS_RING   48
#define DATA_RING       12      

//RGB Streifen
#define NUM_LEDS_STRIP  50 
#define DATA_STRIP      13  

#define LED_TYPE        WS2812
#define COLOR_ORDER GRB 

bool changeMode = false;


CRGB leds_ring[NUM_LEDS_RING];
CRGB leds_strip[NUM_LEDS_STRIP];

// -------------------------------------------------------------

void setup() 
{
  //FastLED initialisieren
  FastLED.addLeds<LED_TYPE, DATA_RING, COLOR_ORDER>(leds_ring, NUM_LEDS_RING)
         .setCorrection(TypicalLEDStrip);
  FastLED.setBrightness(30);      //(0‑255)
  FastLED.clear();
  FastLED.show();

  Serial.begin(9600);
}

// -------------------------------------------------------------

void Blitz(int ringnumber)
{
  int LED_Start;
  int LED_End;
  switch (ringnumber)
  {
    case 1: LED_Start=0; LED_End=16; break;
    case 2: LED_Start=16; LED_End=32; break;
    case 3: LED_Start=32; LED_End=48; break;
    default: break;
  }

  for (uint16_t i = LED_Start; i < LED_End; ++i) 
  {
    leds_ring[i] = CRGB::White;
    FastLED.show();
  }
  delay(500);

  for (uint16_t i = LED_Start; i < LED_End; ++i) 
  {
    leds_ring[i] = CRGB::Black;
    FastLED.show();
  }
}

//Alle LEDs einschalten
void All_ON(void)
{
  for (uint16_t i = 0; i < NUM_LEDS_RING; ++i) 
  {
  leds_ring[i] = CRGB::White;
  FastLED.show();
  }
    
  for (uint16_t i = 0; i < NUM_LEDS_STRIP; ++i) 
  {
  leds_strip[i] = CRGB::White;
  FastLED.show();
  }
}

//Alle LEDs ausschalten
void All_OFF(void)
{
  for (uint16_t i = 0; i < NUM_LEDS_RING; ++i) 
  {
  leds_ring[i] = CRGB::Black;
  FastLED.show();
  }
    
  for (uint16_t i = 0; i < NUM_LEDS_STRIP; ++i) 
  {
  leds_strip[i] = CRGB::Black;
  FastLED.show();
  }
}

void loop() 
{
  /*
  if (Serial.available()) 
  {
    String cmd = Serial.readStringUntil('\n');   // bis Zeilenumbruch lesen
    cmd.trim();                                 // ggf. \r entfernen

    if (cmd.equalsIgnoreCase(F("Change"))) 
    {
      Serial.println(F("Waiting for command (1 = Blitz Ring 1)"));
      changeMode = true;                        // wir gehen jetzt in den Sub‑Modus
    }
    else if (changeMode) 
    {
      char c = cmd.charAt(0);
      switch (c) 
      {
        case '1': Blitz(1); break;
        case '2': Blitz(2); break;
        case '3': Blitz(3); break;
        case 'a': All_ON(); break;
        case '0': All_OFF(); break;
        default:  Serial.println(F("unknown sub‑command")); break;
      }
      changeMode = false;                       // zurück in den Normalmodus
    }
  }
  */
  Blitz(1);

}