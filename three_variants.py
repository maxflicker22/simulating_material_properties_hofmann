# === Imports ===
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === Konstanten ===
quadratmeter = 50
qm_preis = 3000.
kaufpreis = quadratmeter * qm_preis
qm_preis_miete = 12.
miete_start = quadratmeter * qm_preis_miete

nebenkosten_satz = 0.11
zinssatz = 0.03
monatszins = zinssatz / 12
laufzeiten = [15, 20, 30, 40]
farben = ["purple", "green", "blue", "red"]
labels = [
    "Kaufen: 15 Jahre Kredit + ETF",
    "Kaufen: 20 Jahre Kredit + ETF",
    "Kaufen: 30 Jahre Kredit + ETF",
    "Kaufen: 40 Jahre Kredit + ETF"
]

etf_rendite = 0.09
steuer_etf = 0.27
immobilienwertsteigerung = 0.04
mietzinssteigerung = 0.01
vergleichsjahre = 40
vergleichsmonate = vergleichsjahre * 12

# === Kreditberechnungen ===
kreditbetrag = kaufpreis + kaufpreis * nebenkosten_satz
n_monate_max_rate = laufzeiten[0] * 12
rate_max = kreditbetrag * monatszins / (1 - (1 + monatszins) ** -n_monate_max_rate)
gesamtbudget = rate_max

# === ETF + Miete Strategie ===
miete = miete_start
etf_only_list = []
etf_wert = 0
for monat in range(vergleichsmonate):
    miete *= (1 + mietzinssteigerung / 12)
    sparrate = gesamtbudget - miete
    etf_wert *= (1 + etf_rendite * (1 - steuer_etf) / 12)
    etf_wert += sparrate
    etf_only_list.append(etf_wert)

# === Kaufstrategien ===
kaufstrategien = []
for laufzeit, farbe, label in zip(laufzeiten, farben, labels):
    immobilienwert = kaufpreis
    nebenkosten = kaufpreis * nebenkosten_satz
    kreditbetrag_lokal = kaufpreis + nebenkosten
    restschuld = kreditbetrag_lokal

    n_monate = laufzeit * 12
    rate = kreditbetrag_lokal * monatszins / (1 - (1 + monatszins) ** -n_monate)

    etf_kapital = 0
    vermoegen_kaufen_list = []

    for monat in range(vergleichsmonate):
        if monat % 12 == 0:
            immobilienwert *= (1 + immobilienwertsteigerung)

        if monat < n_monate:
            zinsen = restschuld * monatszins
            tilgung = rate - zinsen
            restschuld -= tilgung
            effektive_rate = rate
        else:
            effektive_rate = 0

        etf_rate = gesamtbudget - effektive_rate
        etf_kapital *= (1 + etf_rendite * (1 - steuer_etf) / 12)
        etf_kapital += etf_rate

        vermoegen_kaufen = (immobilienwert - restschuld) + etf_kapital
        vermoegen_kaufen_list.append(vermoegen_kaufen)

    kaufstrategien.append((vermoegen_kaufen_list, label, farbe))

# === Vergleich: 30 Jahre Kredit mit und ohne Eigenkapital ===
eigenkapital_anteil = 0.20
eigenkapital = kaufpreis * eigenkapital_anteil
nebenkosten = kaufpreis * nebenkosten_satz

# Variante A: EK reduziert Kredit
kreditbetrag_A = kaufpreis + nebenkosten - eigenkapital
rate_A = kreditbetrag_A * monatszins / (1 - (1 + monatszins) ** -(30 * 12))
restschuld_A = kreditbetrag_A
etf_A = 0
immo_A = kaufpreis
vermoegen_A = []

# Variante B: EK wird investiert
kreditbetrag_B = kaufpreis + nebenkosten
rate_B = kreditbetrag_B * monatszins / (1 - (1 + monatszins) ** -(30 * 12))
restschuld_B = kreditbetrag_B
etf_B = eigenkapital
immo_B = kaufpreis
vermoegen_B = []

# Einheitliches Budget für beide Varianten (Variante B als Basis)
gesamtbudget_ek = rate_B

for monat in range(vergleichsmonate):
    # Variante A
    if monat % 12 == 0:
        immo_A *= (1 + immobilienwertsteigerung)
        immo_B *= (1 + immobilienwertsteigerung)

    if monat < 30 * 12:
        zinsen_A = restschuld_A * monatszins
        tilgung_A = rate_A - zinsen_A
        restschuld_A -= tilgung_A
        effektive_rate_A = rate_A

        zinsen_B = restschuld_B * monatszins
        tilgung_B = rate_B - zinsen_B
        restschuld_B -= tilgung_B
        effektive_rate_B = rate_B
    else:
        effektive_rate_A = 0
        effektive_rate_B = 0

    # ETF-Verlauf
    etf_rate_A = gesamtbudget_ek - effektive_rate_A
    etf_A *= (1 + etf_rendite * (1 - steuer_etf) / 12)
    etf_A += etf_rate_A
    vermoegen_A.append((immo_A - restschuld_A) + etf_A)

    etf_rate_B = gesamtbudget_ek - effektive_rate_B
    etf_B *= (1 + etf_rendite * (1 - steuer_etf) / 12)
    etf_B += etf_rate_B
    vermoegen_B.append((immo_B - restschuld_B) + etf_B)

# === Plot 1: Hauptvergleich ===
plt.figure(figsize=(12, 6))
plt.plot(range(vergleichsmonate), etf_only_list, label="Nur Miete + ETF", color="orange")
final_values = [etf_only_list[-1]]

for data, label, farbe in kaufstrategien:
    plt.plot(range(vergleichsmonate), data, label=label, color=farbe)
    final_values.append(data[-1])

plt.title("Nominale Vermögensentwicklung über 40 Jahre")
plt.xlabel("Monate")
plt.ylabel("Vermögen (€)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# === Plot 2: 30 Jahre Kredit mit/ohne Eigenkapital ===
plt.figure(figsize=(12, 6))
plt.plot(range(vergleichsmonate), vermoegen_A, label="30 Jahre Kredit mit EK", color="green")
plt.plot(range(vergleichsmonate), vermoegen_B, label="30 Jahre Kredit + EK in ETF", color="blue")
plt.title("Vergleich: Eigenkapital einsetzen oder investieren (30 Jahre Kredit)")
plt.xlabel("Monate")
plt.ylabel("Vermögen (€)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# === Ergebnisvergleich ===
strategienamen = ["Nur Miete + ETF"] + labels
vermoegen_final_nominal_40 = pd.DataFrame({
    "Strategie": strategienamen,
    "Endvermögen nach 40 Jahren (€)": np.round(final_values, 2)
}).sort_values("Endvermögen nach 40 Jahren (€)", ascending=False)

vergleich_ek = pd.DataFrame({
    "Strategie": ["EK reduziert Kredit", "EK in ETF investiert"],
    "Endvermögen nach 40 Jahren (€)": [np.round(vermoegen_A[-1], 2), np.round(vermoegen_B[-1], 2)]
}).sort_values("Endvermögen nach 40 Jahren (€)", ascending=False)

print(vermoegen_final_nominal_40)
print("\n---\n")
print(vergleich_ek)
