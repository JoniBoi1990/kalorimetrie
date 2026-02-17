import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import linregress

if "results" not in st.session_state:
    st.session_state["results"] = None
if "plotbuf" not in st.session_state:
    st.session_state["plotbuf"] = None
if "last_upload_name" not in st.session_state:
    st.session_state["last_upload_name"] = None
if "metrics" not in st.session_state:
    st.session_state["metrics"] = None


def muster_csv():
    df = pd.DataFrame(
        {
            "Zeit_s": np.linspace(0, 1740, 30),
            "Temperatur_C": np.linspace(21.2, 25.2, 30),
            "Phase": ["Vorperiode"] * 10 + ["Hauptperiode"] * 5 + ["Nachperiode"] * 15,
        }
    )
    return df.to_csv(index=False, sep=";", decimal=".")


def parse_uploaded_csv(uploaded_file):
    raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    lines = text.splitlines()
    header_line_idx = next(
        (i for i, line in enumerate(lines) if "time" in line.lower() and "temp" in line.lower()),
        0,
    )

    df_raw = pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep=None,
        engine="python",
        header=header_line_idx,
    )

    df_raw.columns = df_raw.columns.str.replace("\ufeff", "", regex=False).str.strip()
    cols_lower = [c.lower() for c in df_raw.columns]

    has_time = any("time" in c and "(s" in c for c in cols_lower)
    has_temp = any("temp" in c for c in cols_lower)
    has_phase = any("phase" in c for c in cols_lower)
    has_standard = {"zeit_s", "temperatur_c"} <= set(cols_lower)

    if has_time and has_temp and not has_standard:
        df_raw = df_raw.loc[:, df_raw.columns != df_raw.columns[0]]
        if len(df_raw.columns) < 2:
            raise ValueError("Zu wenige Spalten nach Format-Erkennung.")

        rename_map = {df_raw.columns[0]: "Zeit_s", df_raw.columns[1]: "Temperatur_C"}
        df_raw.rename(columns=rename_map, inplace=True)

        if has_phase:
            phase_col = next((c for c in df_raw.columns if "phase" in c.lower()), None)
            if phase_col and phase_col != "Phase":
                df_raw.rename(columns={phase_col: "Phase"}, inplace=True)
        info_text = "CSV erkannt: Temperaturmessungsformat (Datumsspalte ignoriert)."
    elif has_standard:
        if has_phase:
            phase_col = next((c for c in df_raw.columns if "phase" in c.lower()), None)
            if phase_col and phase_col != "Phase":
                df_raw.rename(columns={phase_col: "Phase"}, inplace=True)
        info_text = "CSV erkannt: Kalorimetrieformat (Standard)."
    else:
        raise ValueError("Unbekanntes Format. Bitte Spaltennamen prüfen.")

    required = {"Zeit_s", "Temperatur_C"}
    if not required <= set(df_raw.columns):
        raise ValueError("Benötigte Spalten 'Zeit_s' und 'Temperatur_C' fehlen.")

    for col in ["Zeit_s", "Temperatur_C"]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    df_raw = df_raw.dropna(subset=["Zeit_s", "Temperatur_C"]).reset_index(drop=True)
    if df_raw.empty:
        raise ValueError("Nach dem Einlesen sind keine gültigen Messpunkte vorhanden.")

    return df_raw, info_text


def calculate_calorimetry(df, C_kal, m_loesung, c_loesung, n_stoff):
    idx_pre = df["Phase"] == "Vorperiode"
    idx_post = df["Phase"] == "Nachperiode"
    idx_main = df["Phase"] == "Hauptperiode"

    if idx_pre.sum() < 2 or idx_post.sum() < 2:
        raise ValueError("Vor- und Nachperiode benötigen jeweils mindestens 2 Messpunkte.")
    if idx_main.sum() < 2:
        raise ValueError("Die Hauptperiode benötigt mindestens 2 Messpunkte.")

    slope_pre, intercept_pre, _, _, _ = linregress(df.loc[idx_pre, "Zeit_s"], df.loc[idx_pre, "Temperatur_C"])
    slope_post, intercept_post, _, _, _ = linregress(df.loc[idx_post, "Zeit_s"], df.loc[idx_post, "Temperatur_C"])

    time_main = df.loc[idx_main, "Zeit_s"].values

    def T_pre(t):
        return intercept_pre + slope_pre * t

    def T_post(t):
        return intercept_post + slope_post * t

    areas = []
    for t_split in time_main:
        t_left = time_main[time_main <= t_split]
        t_right = time_main[time_main >= t_split]

        y_left_data = df.loc[idx_main & (df["Zeit_s"] <= t_split), "Temperatur_C"].values
        y_right_data = df.loc[idx_main & (df["Zeit_s"] >= t_split), "Temperatur_C"].values

        area1 = np.trapz(T_pre(t_left) - y_left_data, t_left)
        area2 = np.trapz(y_right_data - T_post(t_right), t_right)
        areas.append(abs(area1 - area2))

    best_idx = int(np.argmin(areas))
    split_time = float(time_main[best_idx])
    T1_corr = float(intercept_pre + slope_pre * split_time)
    T2_corr = float(intercept_post + slope_post * split_time)
    delta_T_corr = T1_corr - T2_corr

    Q_reaktion = (C_kal + m_loesung * c_loesung) * delta_T_corr
    delta_H = Q_reaktion / n_stoff if n_stoff else np.nan

    t_vals = np.linspace(df["Zeit_s"].min(), df["Zeit_s"].max(), 500)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["Zeit_s"], df["Temperatur_C"], "o-", label="Messdaten")
    ax.plot(t_vals, intercept_pre + slope_pre * t_vals, "--", label="Fit Vorperiode")
    ax.plot(t_vals, intercept_post + slope_post * t_vals, "--", label="Fit Nachperiode")
    ax.axvline(split_time, color="red", linestyle="--", label="Equal-Area")

    ax.plot(split_time, T1_corr, "go", label=f"T1 = {T1_corr:.2f} °C")
    ax.plot(split_time, T2_corr, "bo", label=f"T2 = {T2_corr:.2f} °C")
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Temperatur [°C]")
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)

    results = pd.DataFrame(
        {
            "ΔT [°C]": [delta_T_corr],
            "Q [J]": [Q_reaktion],
            "ΔH [kJ/mol]": [delta_H / 1000],
            "C_kal [J/K]": [C_kal],
            "m_Lösung [g]": [m_loesung],
            "c_Lösung [J/g/K]": [c_loesung],
            "n_Stoff [mol]": [n_stoff],
            "Splitzeitpunkt [s]": [split_time],
            "T1 [°C]": [T1_corr],
            "T2 [°C]": [T2_corr],
        }
    )

    metrics = {
        "T1_corr": T1_corr,
        "T2_corr": T2_corr,
        "delta_T_corr": delta_T_corr,
        "Q_reaktion": Q_reaktion,
        "delta_H": delta_H,
    }

    return results, buf.getvalue(), metrics


st.title("Kalorimetrie-Auswertung mit Wärmeaustauschkorrektur")

st.write(
    "Lade eine CSV-Datei hoch, die du aus SparkVue exportiert hast. "
    "Du kannst davor die Spalte 'Phase' wie in der 'Muster'-Datei ergänzen "
    "oder die Phasen nach dem Upload manuell definieren."
)
st.download_button(
    "Muster-CSV herunterladen",
    muster_csv(),
    file_name="kalorimetrie_muster.csv",
    mime="text/csv",
)

uploaded_file = st.file_uploader("CSV-Datei hochladen", type="csv")

C_kal = st.number_input("Kalorimeterkapazität (J/K)", value=120.0)
m_loesung = st.number_input("Masse Lösung (g)", value=100.00)
c_loesung = st.number_input("spez. Wärmekapazität (J/g/K)", value=4.183, format="%.3f")
n_stoff = st.number_input("Stoffmenge (mol)", value=0.002, format="%.3f")
ausgabe_name = st.text_input("Name für die Ausgabe (ohne Endung)", value="kalorimetrie_ergebnis")

if uploaded_file is not None and st.session_state["last_upload_name"] != uploaded_file.name:
    st.session_state["results"] = None
    st.session_state["plotbuf"] = None
    st.session_state["metrics"] = None
    st.session_state["last_upload_name"] = uploaded_file.name

if uploaded_file is not None:
    try:
        df_raw, info_text = parse_uploaded_csv(uploaded_file)
        st.info(info_text)
    except Exception as e:
        st.error(f"Fehler beim Einlesen der CSV-Datei: {e}")
        st.stop()

    manual_phase_mode = "Phase" not in df_raw.columns

    if manual_phase_mode:
        st.warning("Keine Phase-Spalte erkannt. Bitte Phasen manuell definieren.")
        n = len(df_raw)
        col1, col2 = st.columns(2)

        default_vor = max(1, min(n - 2, int(n * 0.3)))
        default_haupt = max(default_vor + 1, min(n - 1, int(n * 0.6)))

        with col1:
            vorperiode_end = st.number_input(
                "Vorperiode endet bei Zeile:",
                min_value=1,
                max_value=n - 1,
                value=default_vor,
                key="vorperiode_end",
            )

        with col2:
            hauptperiode_end = st.number_input(
                "Hauptperiode endet bei Zeile:",
                min_value=int(vorperiode_end) + 1,
                max_value=n,
                value=max(int(vorperiode_end) + 1, default_haupt),
                key="hauptperiode_end",
            )

        df_raw["Phase"] = [
            "Vorperiode" if i < int(vorperiode_end) else "Hauptperiode" if i < int(hauptperiode_end) else "Nachperiode"
            for i in range(n)
        ]

        calculate_clicked = st.button("Manuelle Phasen berechnen", key="manual_calc")
    else:
        calculate_clicked = st.button("Berechnung starten", key="auto_calc")

    if calculate_clicked:
        try:
            results, plot_bytes, metrics = calculate_calorimetry(
                df_raw,
                C_kal,
                m_loesung,
                c_loesung,
                n_stoff,
            )
            st.session_state["results"] = results
            st.session_state["plotbuf"] = plot_bytes
            st.session_state["metrics"] = metrics
        except Exception as e:
            st.error(f"Berechnung nicht möglich: {e}")

if st.session_state.get("metrics") is not None:
    metrics = st.session_state["metrics"]
    st.write(f"**T1_korrigiert:** {metrics['T1_corr']:.2f} °C = {metrics['T1_corr'] + 273.15:.2f} K")
    st.write(f"**T2_korrigiert:** {metrics['T2_corr']:.2f} °C = {metrics['T2_corr'] + 273.15:.2f} K")
    st.write(f"**ΔT:** {metrics['delta_T_corr']:.2f} °C = {metrics['delta_T_corr']:.2f} K")
    st.write(f"**Q:** {metrics['Q_reaktion']:.2f} J")
    st.write(f"**ΔH:** {metrics['delta_H'] / 1000:.2f} kJ/mol")

if st.session_state.get("plotbuf") is not None:
    st.image(st.session_state["plotbuf"], caption="Kalorimetrie-Plot", use_container_width=True)
    st.download_button(
        "Plot als PNG herunterladen",
        st.session_state["plotbuf"],
        file_name=f"{ausgabe_name}.png",
        mime="image/png",
    )

if st.session_state.get("results") is not None:
    st.write("### Ergebnisse")
    st.dataframe(st.session_state["results"], use_container_width=True)
    st.download_button(
        "Ergebnisse als CSV herunterladen",
        st.session_state["results"].to_csv(index=False, decimal=",").encode(),
        file_name=f"{ausgabe_name}.csv",
        mime="text/csv",
    )
