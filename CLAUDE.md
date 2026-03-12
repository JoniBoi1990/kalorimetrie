# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
streamlit run Waerme2.py
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Architecture

This is a Streamlit-based calorimetry analysis tool for chemistry lab experiments. Two versions exist:

- **`Waerme.py`** — original version, simpler, no session state, recalculates on every interaction
- **`Waerme2.py`** — improved version with session state, more robust CSV parsing, manual phase assignment; **this is the current main file**

### Core Algorithm (Equal-Area Criterion)

Both files implement the same calorimetry workflow:

1. **CSV input**: expects columns `Zeit_s` (time in seconds), `Temperatur_C` (temperature in °C), `Phase` (`Vorperiode`/`Hauptperiode`/`Nachperiode`)
2. **Linear regression** on Vor- and Nachperiode via `scipy.stats.linregress`
3. **Equal-area split**: iterates over each time point in Hauptperiode, finds the split time that minimizes `|area_left - area_right|` using `np.trapz`
4. **Corrected temperatures** T1, T2 are read from the regression lines at the split time
5. **Q = (C_kal + m·c) · ΔT** and **ΔH = Q / n**

### `Waerme2.py` specifics

- `parse_uploaded_csv()`: auto-detects CSV format — handles both standard Kalorimetrie format and SparkVue export format (with date column that gets dropped); supports UTF-8 and Latin-1 encoding
- `calculate_calorimetry()`: pure calculation function, returns `(results_df, plot_png_bytes, metrics_dict)`
- Session state keys: `results`, `plotbuf`, `metrics`, `last_upload_name` — results persist across Streamlit reruns until a new file is uploaded
- When no `Phase` column is present, manual row-range inputs are shown to define phase boundaries

## CSV Format

Standard format (semicolon-separated, dot as decimal):
```
Zeit_s;Temperatur_C;Phase
0;21.2;Vorperiode
...
```

SparkVue export format is also accepted (date column auto-dropped, time/temp columns auto-renamed).
