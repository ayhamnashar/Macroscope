# Motor-Steuerung
STEP_SIZE = 100
MOVEMENT_DELAY = 0.5  # s

# --- Z (dein bereits gesteckter Encoder 1) ---
ENCODER_Z_CLK = 17   # Pin 11
ENCODER_Z_DT  = 27   # Pin 13
ENCODER_Z_SW  = 22   # Pin 15

# --- X (Encoder 2) ---
ENCODER_X_CLK = 5    # Pin 29
ENCODER_X_DT  = 6    # Pin 31
ENCODER_X_SW  = 13   # Pin 33

# --- Y (Encoder 3) -> du hast 35/37/40 benutzt ---
ENCODER_Y_CLK = 19   # Pin 35
ENCODER_Y_DT  = 26   # Pin 37
ENCODER_Y_SW  = 21   # Pin 40


# === Kalibrierung (aus deinem Measure-Tool) ===
UM_PER_PX = 1.0   # <- hier deinen ermittelten Wert eintragen (z.B. 0.27, 0.5, 1.0 ...)

# === Zellmaß (für H&E): typischer Kerndurchmesser in µm ===
NUCLEUS_DIAM_UM = 8.0  # 6–10 µm ist üblich

# Aus Kalibrierung abgeleitete Pixelwerte
NUCLEUS_DIAM_PX = max(1.0, NUCLEUS_DIAM_UM / UM_PER_PX)
# Bandpass-Sigmas um die Kerngröße (Faustregeln: d/6 und d/2)
DOG_SIGMA1_PX = max(0.6, NUCLEUS_DIAM_PX / 6.0)
DOG_SIGMA2_PX = max(DOG_SIGMA1_PX + 0.4, NUCLEUS_DIAM_PX / 2.0)

# Tissue-Mask & Score-Gewichte
TISSUE_S_MIN   = 0.20   # HSV-Sättigungsschwelle (0.15–0.25 testen)
TISSUE_V_MAX   = 0.97   # sehr helle Highlights raus
NUCLEI_WEIGHT  = 0.35   # Boost für bläuliche Kerne (0.2–0.6)
COVERAGE_ALPHA = 0.7    # Score-Skalierung nach Tissue-Deckung (0.5–1.0)
