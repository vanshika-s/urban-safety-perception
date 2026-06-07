# Urban Safety Perception - San Diego

**DSC 148 - Introduction to Data Mining**
By Vanshika Somani

---

## Overview

This project predicts whether a San Diego location is safe or unsafe using environmental features: EPA walkability scores, SDPD crime density, streetlight density, and geographic context. A machine learning model (XGBoost) is trained on a grid of 7,872 San Diego locations and deployed as an interactive web app.

**Live demo:** https://urban-safety-perception.streamlit.app/

---

## Research Question

Can machine learning predict the perceived safety of a San Diego location using publicly available environmental data?

---

## Data Sources

| Dataset | Source | Size |
|---|---|---|
| SDPD Calls for Service (2024) | [San Diego Open Data](https://data.sandiego.gov/datasets/police-calls-for-service/) | 970,650 incidents |
| EPA National Walkability Index | [Kaggle — stacey06/u-s-walkability-index](https://www.kaggle.com/datasets/stacey06/u-s-walkability-index) | 3,443 SD block groups |
| SD Streetlight Locations | [SD ArcGIS REST API](https://webmaps.sandiego.gov/) | 56,058 lights |
| Census TIGER Block Groups (CA) | [Census Bureau](https://www2.census.gov/geo/tiger/TIGER2020/BG/tl_2020_06_bg.zip) | CA block groups |

---

## Methodology

### Feature Engineering

- **Crime score** - beat-level crime density spatially joined from SDPD beat boundaries, normalized and inverted (higher = safer). 120 unique values across 134 beats.
- **Walkability score** - EPA NatWalkInd normalized to 0-1
- **Lighting score** - streetlight count within 200m radius, normalized to 0–1
- **Composite safety score** - weighted formula: `0.50 × crime + 0.25 × walkability + 0.25 × lighting`

### Models

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 0.980 | 0.984 | 1.000 |
| Random Forest | 0.994 | 0.995 | 1.000 |
| **XGBoost** | **0.996** | **0.997** | **1.000** |

### Ablation Study (XGBoost)

| Features | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Walkability only | 0.843 | 0.877 | 0.869 |
| Crime only | 0.707 | 0.797 | 0.771 |
| Lighting only | 0.596 | 0.739 | 0.548 |
| All features | 0.996 | 0.997 | 1.000 |

**Key finding:** Walkability and crime score are co-dominant predictors (mean |SHAP| 5.4 and 5.1 respectively), with lighting providing secondary signal.

---

## Project Structure

```
urban-safety-perception/
├── data/
│   ├── raw/                    # Raw datasets (not tracked in git)
│   └── processed/              # modeling_df.csv, results, plots
├── notebooks/
│   ├── 01_preprocessing.ipynb  # Data cleaning & feature engineering
│   └── 02_models.ipynb         # Model training, evaluation, SHAP
├── app/
│   └── app.py                  # Streamlit demo app
├── requirements.txt
└── README.md
```

---

## Setup & Running

```bash
# Clone repo
git clone https://github.com/vanshika-s/urban-safety-perception.git
cd urban-safety-perception

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

### Data Downloads Required

Place these in `data/raw/` before running notebooks:

- `crime_2024.csv` - from San Diego Open Data
- `walkability.csv` - EPA National Walkability Index (Kaggle)
- `pd_beats_datasd.geojson` - SDPD beat boundaries (San Diego Open Data)
- `census_bg/` - Census TIGER Block Groups CA
- `streetlights_full.geojson` - auto-fetched from SD ArcGIS API on first run

---

## Limitations

- Crime score is aggregated at the SDPD beat level (134 beats), with 58% of grid points falling into one large central beat. Finer-grained geocoordinate crime data would improve spatial resolution.
- Safety labels are derived from a weighted formula rather than human perception surveys, so high model accuracy reflects formula recovery rather than generalization to real perceived safety.

---

## Acknowledgements

- [SafePath](https://safepath.up.railway.app/) (DS3 @ UCSD) - scoring formula inspiration
- EPA Smart Location Database
- City of San Diego Open Data Portal
