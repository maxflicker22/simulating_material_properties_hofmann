import numpy as np

quadratmeter = 50
qm_preis = 3000.  # €/qm
kaufpreis = qm_preis * quadratmeter  # Kaufpreis in €
qm_preis_miete = 12.
miete = qm_preis_miete * quadratmeter  # Miete in €/qm

# === Eingaben ===
kaufpreis = kaufpreis  # €
nebenkosten_satz = 0.11
nebenkosten = kaufpreis * nebenkosten_satz
kreditbetrag = kaufpreis + nebenkosten  # Kredit = Kaufpreis + Nebenkosten

zinssatz = 0.03      # 4 % fix
laufzeit_jahre = 30
monatszins = zinssatz / 12
n_monate = laufzeit_jahre * 12


# === Kreditrate berechnen ===
rate = kreditbetrag * monatszins / (1 - (1 + monatszins) ** -n_monate)

gesamtbudget = rate     # €/Monat bei beiden Varianten
miete = miete            # €/Monat bei Miete
etf_rendite = 0.04  # 4 % p.a. Rendite
vergleichsjahre = laufzeit_jahre
vergleichsmonate = vergleichsjahre * 12
immobilienwert = kaufpreis  # bleibt konstant
immobilenwertsteigerung = 0.04
mietzinssteigerung = 0.01

# === Variante 1: Kaufen ===
restschuld = kreditbetrag
getilgter_betrag = 0
gezahlt_zinsen = 0

vermoegen_kaufen_list = []
vermoegen_verkauft_reingewinn = []

for monat in range(vergleichsmonate):
    if monat % 12 == 0:
        immobilienwert *= (1 + immobilenwertsteigerung)

    zinsen = restschuld * monatszins
    tilgung = rate - zinsen
    restschuld -= tilgung
    getilgter_betrag += tilgung
    gezahlt_zinsen += zinsen
    print("restschuld", restschuld)
    print("immobilienwert", immobilienwert)
    vermoegen_kaufen_list.append(immobilienwert - restschuld)
    print("immobilienweert - restschuld", immobilienwert - restschuld)
    vermoegen_verkauft_reingewinn.append((immobilienwert - restschuld) * 0.7)

vermoegen_kaufen = immobilienwert - restschuld

# === Variante 2: Mieten + ETF ===
sparrate = gesamtbudget - miete
etf_wert = 0
etf_wert_list = []

for monat in range(vergleichsmonate):
    miete *= (1 + (mietzinssteigerung/12))
    sparrate = gesamtbudget - miete
    etf_wert *= (1 + etf_rendite / 12)
    etf_wert += sparrate
    etf_wert_list.append(etf_wert)

# === Ausgabe ===
print("🔷 Variante 1: Kaufen")
print(f"- Kreditbetrag inkl. Nebenkosten: {kreditbetrag:.2f} €")
print(f"- Monatliche Rate:                {rate:.2f} €")
print(f"- Getilgt nach 5 Jahren:          {getilgter_betrag:.2f} €")
print(f"- Restschuld:                     {restschuld:.2f} €")
print(f"- Vermögen (Immobilie - Restschuld): {vermoegen_kaufen:.2f} €")

print("\n🔷 Variante 2: Mieten + ETF")
print(f"- Monatliche Miete:            {miete:.2f} €")
print(f"- Monatliche Sparrate:            {sparrate:.2f} €")
print(f"- ETF-Wert nach 5 Jahren:         {etf_wert:.2f} €")

# Vergleich
diff = vermoegen_kaufen - etf_wert
if diff > 0:
    print(f"\n🏡 Vorteil Kaufen: Du hast {diff:.2f} € mehr Vermögen.")
else:
    print(f"\n📈 Vorteil ETF: Du hast {-diff:.2f} € mehr Vermögen.")


# Plotten der Ergebnisse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.dates as mdates
import matplotlib.dates as mdates

differenc = np.array(vermoegen_kaufen_list) - np.array(etf_wert_list)
plt.plot(range(vergleichsmonate), vermoegen_kaufen_list, label="Kaufen", color="blue")
plt.plot(range(vergleichsmonate), etf_wert_list, label="Mieten + ETF", color="orange")
plt.plot(range(vergleichsmonate), vermoegen_verkauft_reingewinn, label="Vermögen Verkauft", color="red")
plt.plot(range(vergleichsmonate), differenc, label="Vermögen Differenz ", color="green")

plt.title("Vermögensentwicklung: Kaufen vs. Mieten + ETF")
plt.xlabel("Monate")
plt.ylabel("Vermögen (€)")
plt.legend()
plt.grid()
#plt.ylim(bottom=0,top=400000)

plt.show()

vergleichsmonate_neu = 1 * vergleichsmonate

etf_wert_von_null = 0
etf_wert_von_null_list = []
# Weiter in die Zukunft
for monat in range(vergleichsmonate_neu):
    etf_wert *= (1 + etf_rendite / 12)
    etf_wert += sparrate
    etf_wert_list.append(etf_wert)

    etf_wert_von_null *= (1 + etf_rendite / 12)
    etf_wert_von_null += rate
    etf_wert_von_null_list.append(etf_wert_von_null)

etf_wert_von_null_list = np.array(etf_wert_von_null_list) + vermoegen_kaufen
plt.plot(range(vergleichsmonate_neu), etf_wert_von_null_list, label="Kaufen", color="blue")
plt.plot(range(vergleichsmonate_neu), etf_wert_list[len(range(vergleichsmonate)):], label="Mieten + ETF", color="orange")

plt.title("Vermögensentwicklung: Kaufen vs. Mieten + ETF")
plt.xlabel("Monate")
plt.ylabel("Vermögen (€)")
plt.legend()
plt.grid()
#plt.ylim(bottom=0,top=400000)

plt.show()