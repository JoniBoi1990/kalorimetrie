import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import io

if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'plotbuf' not in st.session_state:
    st.session_state['plotbuf'] = None

def muster_csv():
    df = pd.DataFrame({
        "Zeit_s": np.linspace(0, 1740, 30),
        "Temperatur_C": np.linspace(21.2, 25.2, 30),
        "Phase": ["Vorperiode"]*10 + ["Hauptperiode"]*5 + ["Nachperiode"]*15
    })
    return df.to_csv(index=False, sep=';', decimal='.')

st.title("Kalorimetrie-Auswertung mit Wärmeaustauschkorrektur")

st.write("Lade eine CSV-Datei hoch, die du aus SparkVue exportiert hast. Du kannst davor die Spalte 'Phase' wie in der 'Muster'-Datei ergänzen oder die Phasen nach dem Upload manuell definieren.")
st.download_button("Muster-CSV herunterladen", muster_csv(), file_name="kalorimetrie_muster.csv", mime="text/csv")

uploaded_file = st.file_uploader("CSV-Datei hochladen", type="csv")

# NEU: Eingabeparameter
C_kal = st.number_input("Kalorimeterkapazität (J/K)", value=120.0)
m_loesung = st.number_input("Masse Lösung (g)", value=100.00)
c_loesung = st.number_input("spez. Wärmekapazität (J/g/K)", value=4.183, format="%.3f")
n_stoff = st.number_input("Stoffmenge (mol)", value=0.002, format="%.3f")
ausgabe_name = st.text_input("Name für die Ausgabe (ohne Endung)", value="kalorimetrie_ergebnis")

if uploaded_file and st.button("Berechnung starten"):

    # --- Einlesen der CSV mit automatischer Headererkennung ---
    try:
        lines = uploaded_file.read().decode("utf-8").splitlines()
        header_line_idx = next(
            (i for i, line in enumerate(lines) if "Time" in line and "Temp" in line),
            0
        )
        from io import StringIO
        df_raw = pd.read_csv(StringIO("\n".join(lines)),
                             sep=None, engine="python", header=header_line_idx)
    except Exception as e:
        st.error(f"Fehler beim Einlesen der CSV-Datei: {e}")
        st.stop()

    # --- Spaltennamen bereinigen und BOM entfernen ---
    df_raw.columns = df_raw.columns.str.replace('\ufeff', '', regex=False).str.strip()

    cols_lower = [c.lower() for c in df_raw.columns]

    # --- Erkennung der Formate und 'Phase'-Spalte ---
    has_time = any("time" in c and "(s" in c for c in cols_lower)
    has_temp = any("temp" in c for c in cols_lower)
    has_phase = any("phase" in c for c in cols_lower)
    has_standard = {"zeit_s", "temperatur_c"} <= set(cols_lower)

    if has_time and has_temp and not has_standard:
        # Mit Datumsspalte
        df_raw = df_raw.loc[:, df_raw.columns != df_raw.columns[0]]  # Entfernt nur erste Spalte (Datum)
        
        # Benenne die ersten beiden Spalten um (Zeit_s, Temperatur_C)
        rename_map = {df_raw.columns[0]: "Zeit_s", df_raw.columns[1]: "Temperatur_C"}
        df_raw.rename(columns=rename_map, inplace=True)

        if has_phase:
            phase_col = next((c for c in df_raw.columns if "phase" in c.lower()), None)
            if phase_col and phase_col != "Phase":
                df_raw.rename(columns={phase_col: "Phase"}, inplace=True)
        st.info("CSV erkannt: Temperaturmessungsformat – Datumsspalte ignoriert.")
    elif has_standard:
        # Standard-Kalorimetrie-Format
        st.info("CSV erkannt: Kalorimetrieformat (Standard).")
        if has_phase:
            phase_col = next((c for c in df_raw.columns if "phase" in c.lower()), None)
            if phase_col and phase_col != "Phase":
                df_raw.rename(columns={phase_col: "Phase"}, inplace=True)
    else:
        st.warning("Unbekanntes Format. Bitte Spaltennamen prüfen.")
        st.stop()

    # --- Wenn keine Phase-Spalte, manuelle Eingabe ---
    if "Phase" not in df_raw.columns:
        st.warning("Keine Phase-Spalte erkannt. Bitte Phasen manuell definieren.")
        n = len(df_raw)
        col1, col2, col3 = st.columns(3)
        with col1:
            vorperiode_end = st.number_input("Vorperiode endet bei Zeile:",
                                            min_value=1, max_value=n,
                                            value=int(n * 0.3),
                                            key="vorperiode_end")
        
        with col2:
            hauptperiode_end = st.number_input("Hauptperiode endet bei Zeile:",
                                              min_value=int(vorperiode_end) + 1,
                                              max_value=n,
                                              value=int(n * 0.6),
                                              key="hauptperiode_end")
        df_raw["Phase"] = [
            "Vorperiode" if i < vorperiode_end else
            "Hauptperiode" if i < hauptperiode_end else
            "Nachperiode"
            for i in range(n)
        ]
        # Hier ergänzen wir den neuen Button für manuelle Phasenberechnung:
        if st.button("Manuelle Phasen berechnen"):
            # Hier DIE Berechnungslogik AUFRUFEN,
            # z.B. Funktionsaufruf (eigene Berechnung)
            results, plot_bytes = do_calculation(df_raw, ...)

            # Ergebnisse und Plot im Session State speichern (für stabile Anzeige)
            st.session_state['results'] = results
            st.session_state['plotbuf'] = plot_bytes

    else:
        # Wenn Phase-Spalte da ist, normale Berechnung starten
        results, plot_bytes = do_calculation(df_raw, ...)
        st.session_state['results'] = results
        st.session_state['plotbuf'] = plot_bytes

    if st.session_state.get('results') is not None:
        st.write("### Ergebnisse")
        st.write(st.session_state['results'])  # Zeigt die berechneten Werte an

        if st.session_state.get('plotbuf') is not None:
            st.image(st.session_state['plotbuf'], caption="Kalorimetrie-Plot", use_column_width=True)


    # Ab hier geht deine bisherige Analyse weiter,
    # z.B. Umbenennung für Einheitlichkeit, Berechnungen etc.

    df = df_raw.copy()

    idx_pre = df['Phase'] == "Vorperiode"
    idx_post = df['Phase'] == "Nachperiode"
    idx_main = df['Phase'] == "Hauptperiode"

    # Lineare Regressionen
    slope_pre, intercept_pre, _, _, _ = linregress(df.loc[idx_pre, 'Zeit_s'], df.loc[idx_pre, 'Temperatur_C'])
    slope_post, intercept_post, _, _, _ = linregress(df.loc[idx_post, 'Zeit_s'], df.loc[idx_post, 'Temperatur_C'])

    time_main = df.loc[idx_main, 'Zeit_s'].values

    def T_pre(t): return intercept_pre + slope_pre * t
    def T_post(t): return intercept_post + slope_post * t

    # Equal-Area-Kriterium
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

    # Debug-Ausgabe
    # st.write("### DEBUG")
    # st.write(f"Splitzeitpunkt (s): {split_time}")
    # st.write(f"Steigung Vorperiode: {slope_pre:.7f} °C/s – Intercept: {intercept_pre:.4f}")
    # st.write(f"Steigung Nachperiode: {slope_post:.7f} °C/s – Intercept: {intercept_post:.4f}")
    # st.write(f"Korrigierte ΔT = {delta_T_corr:.4f} °C")

    # Plot
    t_vals = np.linspace(df['Zeit_s'].min(), df['Zeit_s'].max(), 500)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['Zeit_s'], df['Temperatur_C'], 'o-', label='Messdaten')
    ax.plot(t_vals, intercept_pre + slope_pre * t_vals, '--', label='Fit Vorperiode')
    ax.plot(t_vals, intercept_post + slope_post * t_vals, '--', label='Fit Nachperiode')
    ax.axvline(split_time, color='red', linestyle='--', label='Equal-Area')

    ax.plot(0, T1_corr, 'go', label=f"T1 = {T1_corr:.2f} °C")
    ax.plot(0, T2_corr, 'bo', label=f"T2 = {T2_corr:.2f} °C")
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Temperatur [°C]")
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    st.pyplot(fig)
    st.download_button("Plot als PNG herunterladen", buf.getvalue(), file_name=f"{ausgabe_name}.png", mime="image/png")

    st.write(f"**T1_korrigiert:** {T1_corr:.2f} °C = {T1_corr + 273.15:.2f} K")
    st.write(f"**T2_korrigiert:** {T2_corr:.2f} °C = {T2_corr + 273.15:.2f} K")
    st.write(f"**ΔT:** {delta_T_corr:.2f} °C = {delta_T_corr + 273.15:.2f} K")
    st.write(f"**Q:** {Q_reaktion:.2f} J")
    st.write(f"**ΔH:** {delta_H / 1000:.2f} kJ/mol")

    results = pd.DataFrame({
        "ΔT [°C]": [delta_T_corr],
        "Q [J]": [Q_reaktion],
        "ΔH [kJ/mol]": [delta_H / 1000],
        "C_kal [J/K]": [C_kal],
        "m_Lösung [g]": [m_loesung],
        "c_Lösung [J/g/K]": [c_loesung],
        "n_Stoff [mol]": [n_stoff],
        "Splitzeitpunkt [s]": [split_time],
        "T1 [°C]": [T1_corr],
        "T2 [°C]": [T2_corr]
    })

    st.download_button("Ergebnisse als CSV herunterladen", results.to_csv(index=False, decimal=',').encode(), file_name=f"{ausgabe_name}.csv", mime="text/csv")
