import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import io

def muster_csv():
    df = pd.DataFrame({
        "Zeit_s": np.linspace(0, 1740, 30),
        "Temperatur_C": np.linspace(21.2, 25.2, 30),
        "Phase": ["Vorperiode"]*10 + ["Hauptperiode"]*5 + ["Nachperiode"]*15
    })
    return df.to_csv(index=False, sep=';', decimal=',')

st.title("Kalorimetrie-Auswertung mit Wärmeaustauschkorrektur")

st.write("1. Lade eine eigene CSV mit den Spalten **Zeit_s**, **Temperatur_C**, **Phase** (Vorperiode/Hauptperiode/Nachperiode) hoch.")
st.download_button("Muster-CSV herunterladen", muster_csv(), file_name="kalorimetrie_muster.csv", mime="text/csv")

uploaded_file = st.file_uploader("CSV-Datei hochladen", type="csv")
C_kal = st.number_input("Kalorimeterkapazität (J/K)", value=120.0)
m_loesung = st.number_input("Masse Lösung (kg)", value=0.05)
c_loesung = st.number_input("spez. Wärmekapazität (J/kg/K)", value=4180)
n_stoff = st.number_input("Stoffmenge (mol)", value=0.002, format="%.3f")
ausgabe_name = st.text_input("Name für die Ausgabe (ohne Endung)", value="kalorimetrie_ergebnis")

if uploaded_file and st.button("Berechnung starten"):
    df = pd.read_csv(uploaded_file, decimal=',', sep=';')
    st.write("Vorschau der Daten:")
    st.dataframe(df.head())

    idx_pre = df['Phase'] == "Vorperiode"
    idx_post = df['Phase'] == "Nachperiode"
    idx_main = df['Phase'] == "Hauptperiode"

    slope_pre, intercept_pre, r_pre, _, _ = linregress(df.loc[idx_pre, 'Zeit_s'], df.loc[idx_pre, 'Temperatur_C'])
    slope_post, intercept_post, r_post, _, _ = linregress(df.loc[idx_post, 'Zeit_s'], df.loc[idx_post, 'Temperatur_C'])

    time_main = df.loc[idx_main, 'Zeit_s'].values

    def T_pre(t): return intercept_pre + slope_pre * t
    def T_post(t): return intercept_post + slope_post * t

    areas = []
    for t_split in time_main:
        area1 = np.trapz(
            T_pre(time_main[time_main <= t_split]) -
            df.loc[idx_main & (df['Zeit_s'] <= t_split), 'Temperatur_C'].values,
            time_main[time_main <= t_split])
        area2 = np.trapz(
            df.loc[idx_main & (df['Zeit_s'] >= t_split), 'Temperatur_C'].values -
            T_post(time_main[time_main >= t_split]),
            time_main[time_main >= t_split])
        areas.append(abs(area1 - area2))

    best_idx = np.argmin(areas)
    split_time = time_main[best_idx]

    T1_corr = intercept_pre + slope_pre * split_time
    T2_corr = intercept_post + slope_post * split_time
    delta_T_corr = T1_corr - T2_corr

    Q_reaktion = (C_kal + m_loesung * c_loesung) * delta_T_corr
    delta_H = Q_reaktion / n_stoff if n_stoff else np.nan

    # Debug-Ausgabe (Achsenabschnitt und Steigung)
    st.write("### DEBUG-Ausgaben")
    st.write(f"Splitzeitpunkt (s): {split_time}")
    st.write(f"Interzept Vorperiode: {intercept_pre:.4f} °C")
    st.write(f"Steigung Vorperiode: {slope_pre:.7f} °C/s")
    st.write(f"Interzept Nachperiode: {intercept_post:.4f} °C")
    st.write(f"Steigung Nachperiode: {slope_post:.7f} °C/s")
    st.write(f"Korrigierte Temperatur T1 (°C): {T1_corr:.4f}")
    st.write(f"Korrigierte Temperatur T2 (°C): {T2_corr:.4f}")
    st.write(f"Berechnete Temperaturdifferenz ΔT (°C): {delta_T_corr:.4f}")

    t_vals = np.linspace(df['Zeit_s'].min(), df['Zeit_s'].max(), 500)
    T_vor = intercept_pre + slope_pre * t_vals
    T_nach = intercept_post + slope_post * t_vals

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['Zeit_s'], df['Temperatur_C'], 'o-', label='Messwerte')
    ax.plot(t_vals, T_vor, '--', label='Fit Vorperiode')
    ax.plot(t_vals, T_nach, '--', label='Fit Nachperiode')
    ax.axvline(split_time, color='red', linestyle='--', label='Equal-Area-Kriterium')

    # Keine horizontalen Linien mehr für korrigierte Temperaturen

    # Stattdessen Markierungen an der y-Achse (x=0) für korrigierte Temperaturen
    ax.plot(0, T1_corr, 'go', label='T1 korrigiert')
    ax.plot(0, T2_corr, 'bo', label='T2 korrigiert')
    ax.text(0, T1_corr, f'{T1_corr:.2f} °C', color='green', ha='right', va='center')
    ax.text(0, T2_corr, f'{T2_corr:.2f} °C', color='blue', ha='right', va='center')

    ax.set_xlabel('Zeit [s]')
    ax.set_ylabel('Temperatur [°C]')
    ax.set_title('Kalorimetrische Messung mit Korrektur')
    ax.legend()
    plt.tight_layout()


    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    st.pyplot(fig)
    st.download_button("Plot als PNG herunterladen", buf.getvalue(), file_name=f"{ausgabe_name}.png", mime='image/png')

    # Nur korrigierte Temperaturen ausgeben, keine Achsenabschnitte
    st.write(f"**Korrigierte Temperaturdifferenz ΔT:** {delta_T_corr:.3f} °C")
    st.write(f"**Reaktionswärme Q:** {Q_reaktion:.2f} J")
    st.write(f"**Reaktionsenthalpie ΔH:** {delta_H / 1000:.2f} kJ/mol")

    st.write("**Korrigierte Temperaturen:**")
    st.write(f"- Vorperiode (T1): {T1_corr:.3f} °C")
    st.write(f"- Nachperiode (T2): {T2_corr:.3f} °C")

    results = pd.DataFrame({
        "Korrigierte_Temperaturdifferenz_GradC": [delta_T_corr],
        "Reaktionswaerme_J": [Q_reaktion],
        "Reaktionsenthalpie_kJ_mol": [delta_H / 1000],
        "C_kal_J_K": [C_kal],
        "m_Loesung_kg": [m_loesung],
        "c_Loesung_J_kg_K": [c_loesung],
        "n_Stoff_mol": [n_stoff],
        "Zeitpunkt_Korrektur_s": [split_time],
        "T1_korr_GradC": [T1_corr],
        "T2_korr_GradC": [T2_corr]
    })
    st.download_button("Ergebnisse als CSV herunterladen", results.to_csv(index=False, decimal=',').encode(),
                       file_name=f"{ausgabe_name}.csv", mime='text/csv')
