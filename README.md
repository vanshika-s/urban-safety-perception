# Urban Safety Perception - San Diego

**DSC 148 - Introduction to Data Mining | UC San Diego**
By Vanshika Somani

---

## Overview

This project predicts whether a San Diego location is perceived as safe or unsafe using environmental features - EPA walkability scores, streetlight density, and geographic context. A machine learning model (XGBoost) is trained on a grid of 7,872 San Diego locations and deployed as an interactive web app.

**Live demo:** [Launch App](https://github.com/vanshika-s/urban-safety-perception)

---

## Research Question

> Can machine learning predict the perceived safety of a San Diego location using publicly available environmental data?

---

## Data Sources

| Dataset | Source | Size |
|---|---|---|
| SDPD Calls for Service (2023) | [San Diego Open Data](https://data.sandiego.gov/datasets/police-calls-for-service/) | 463k incidents |
| EPA National Walkability Index | [EPA Smart Location Mapping](https://www.epa.gov/smartgrowth/smart-location-mapping) | 3,443 SD block groups |
| SD Streetlight Locations | [SD ArcGIS REST API](https://webmaps.sandiego.gov/) | 56,058 lights |
| Census TIGER Block Groups (CA) | [Census Bureau](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | CA block groups |

---

## Methodology

### Feature Engineering
- **Crime score** - beat-level crime density normalized and inverted (higher = safer)
- **Walkability score** - EPA NatWalkInd normalized to 0–1
- **Lighting score** - streetlight count within 200m radius, normalized to 0–1
- **Composite safety score** - weighted formula: `0.50 × crime + 0.25 × walkability + 0.25 × lighting`

### Models
| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 0.978 | 0.984 | 1.000 |
| Random Forest | 0.999 | 1.000 | 1.000 |
| **XGBoost** | **0.999** | **1.000** | **1.000** |

### Ablation Study (XGBoost)
| Features | Accuracy | AUC-ROC |
|---|---|---|
| Walkability only | 0.979 | 0.997 |
| Lighting only | 0.683 | 0.639 |
| All features | 0.999 | 1.000 |

**Key finding:** Walkability is the dominant predictor of perceived urban safety (SHAP weight: 0.70).

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
- `crime_2023.csv` - from San Diego Open Data
- `walkability.csv` - EPA National Walkability Index (Kaggle)
- `census_bg/` - Census TIGER Block Groups CA (auto-downloaded)
- `streetlights_full.geojson` - auto-fetched from SD ArcGIS API on first run

---

## Limitations

- Crime score has no spatial variation (beat-level median assigned uniformly) due to missing geocoordinates in raw SDPD data
- Safety labels are derived from a weighted formula rather than human perception surveys
- Model performance is high because labels are a deterministic function of features

---

## Report

[View Report PDF](#) - link to be added after submission

---

## Acknowledgements

- SafePath (DS3 @ UCSD) - scoring formula inspiration
- EPA Smart Location Database
- City of San Diego Open Data Portal
