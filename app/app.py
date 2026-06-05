import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
from shapely.geometry import Point
from xgboost import XGBClassifier
import requests

st.set_page_config(
    page_title="Urban Safety Perception — San Diego",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], p, div, span, label, input {
    font-family: 'Inter', sans-serif !important;
}
.stApp { background-color: #ffffff; }
h1,h2,h3,h4 { font-family:'Inter',sans-serif !important; color:#1a2e4a; font-weight:600; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
.block-container { padding-top:1.5rem !important; padding-bottom:0 !important; max-width:100% !important; }

/* Remove ALL Streamlit container borders/backgrounds */
[data-testid="column"] > div > div > div > div { background:transparent !important; }

/* Landing */
.l-title { font-size:2.5rem; font-weight:700; color:#1a2e4a; text-align:center; margin-bottom:0.3rem; }
.l-sub { font-size:1rem; color:#6c7a8d; text-align:center; margin-bottom:1.5rem; }
.l-card {
    background:#f7f9fc; border:1px solid #e2e8f0; border-radius:14px;
    padding:1.6rem 2rem; max-width:560px; width:100%; margin:1.5rem auto 0 auto;
}
.mrow { display:flex; gap:0.75rem; padding:0.45rem 0; border-bottom:1px solid #edf0f4; font-size:0.9rem; color:#4a5568; align-items:center; }
.mrow:last-child { border-bottom:none; }
.mlbl { font-weight:600; color:#1a2e4a; min-width:80px; }
.l-link { color:#3a7abf; text-decoration:none; font-weight:500; }
.l-about { max-width:560px; width:100%; margin:1.2rem auto 0 auto; font-size:0.88rem; color:#6c7a8d; line-height:1.8; text-align:center; }

/* Left panel — NO box, just background color on the column */
.panel-wrap { background:#ddeeff; border-radius:12px; padding:1.2rem; height:100%; }
.panel-title { font-size:1.05rem; font-weight:700; color:#1a2e4a; margin-bottom:0; }
.divider { border:none; border-top:1px solid #c8dff0; margin:0.8rem 0; }
.result-safe {
    background:#d4edda; border-left:4px solid #28a745; border-radius:8px;
    padding:12px 14px; color:#155724; font-weight:700; font-size:0.95rem; margin:0.6rem 0;
}
.result-unsafe {
    background:#f8d7da; border-left:4px solid #dc3545; border-radius:8px;
    padding:12px 14px; color:#721c24; font-weight:700; font-size:0.95rem; margin:0.6rem 0;
}
.metric-row { display:flex; gap:0.4rem; margin:0.5rem 0; }
.metric-box { flex:1; background:white; border-radius:8px; padding:8px 10px; text-align:center; }
.metric-val { font-size:1.1rem; font-weight:700; color:#1a2e4a; }
.metric-lbl { font-size:0.7rem; color:#6c7a8d; margin-top:2px; }

/* Buttons */
.stButton > button {
    background-color:#3a7abf !important; color:white !important;
    border:none !important; border-radius:8px !important;
    font-family:'Inter',sans-serif !important; font-weight:600 !important;
    outline:none !important; box-shadow:none !important;
}
.stButton > button:hover { background-color:#2c5f94 !important; }
.stButton > button:focus { outline:none !important; box-shadow:none !important; border:none !important; }

/* Inputs */
.stTextInput > div > div > input {
    border-radius:8px !important; border:1px solid #c8dff0 !important;
    font-family:'Inter',sans-serif !important; background:white !important;
}
.stNumberInput > div > div > input {
    border-radius:8px !important; border:1px solid #c8dff0 !important;
    font-family:'Inter',sans-serif !important;
}

/* Make map fill full height with no gap */
[data-testid="stVerticalBlock"] iframe {
    display:block;
}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / 'data' / 'processed'
RAW = ROOT / 'data' / 'raw'

SD_LAT_MIN, SD_LAT_MAX = 32.53, 33.11
SD_LON_MIN, SD_LON_MAX = -117.30, -116.08

def is_in_sd(lat, lon):
    return SD_LAT_MIN <= lat <= SD_LAT_MAX and SD_LON_MIN <= lon <= SD_LON_MAX

def geocode_address(address):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={'q': address + ', San Diego, CA', 'format': 'json', 'limit': 1},
            headers={'User-Agent': 'urban-safety-dsc148'}, timeout=5
        )
        res = r.json()
        if res:
            return float(res[0]['lat']), float(res[0]['lon'])
    except Exception:
        pass
    return None, None

@st.cache_data
def load_modeling_df():
    return pd.read_csv(PROCESSED / 'modeling_df.csv')

@st.cache_resource
def train_model(df):
    X = df[['walk_score', 'light_score', 'lat', 'lon']]
    y = df['safe_label']
    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    model.fit(X, y)
    return model

@st.cache_data
def load_census_walk():
    shp = list((RAW / 'census_bg').glob('*.shp'))[0]
    census = gpd.read_file(shp)
    census_sd = census[census['COUNTYFP'] == '073'].copy().to_crs('EPSG:4326')
    walk = pd.read_csv(RAW / 'walkability.csv', low_memory=False)
    walk_sd = walk[walk['COUNTYFP'] == 73].copy()
    walk_sd['GEOID_fixed'] = (
        '06' + walk_sd['COUNTYFP'].astype(str).str.zfill(3) +
        walk_sd['TRACTCE'].astype(str).str.zfill(6) +
        walk_sd['BLKGRPCE'].astype(str).str.zfill(1)
    )
    walk_sd['walk_score'] = (
        walk_sd['NatWalkInd'] - walk_sd['NatWalkInd'].min()
    ) / (walk_sd['NatWalkInd'].max() - walk_sd['NatWalkInd'].min())
    census_sd['GEOID'] = census_sd['GEOID'].astype(str)
    return census_sd.merge(walk_sd[['GEOID_fixed', 'walk_score']],
                           left_on='GEOID', right_on='GEOID_fixed', how='left')

@st.cache_data
def load_lights():
    return gpd.read_file(RAW / 'streetlights_full.geojson').to_crs('EPSG:32611')

def get_features(lat, lon, census_walk, lights_proj):
    pt = gpd.GeoDataFrame({'geometry': [Point(lon, lat)]}, crs='EPSG:4326')
    joined = gpd.sjoin(pt, census_walk[['geometry', 'walk_score']],
                       how='left', predicate='within')
    walk_score = (float(joined['walk_score'].iloc[0])
                  if not joined['walk_score'].isna().iloc[0] else 0.35)
    pt_proj = pt.to_crs('EPSG:32611')
    buf = pt_proj.geometry.iloc[0].buffer(200)
    light_count = int(lights_proj.geometry.within(buf).sum())
    light_score = min(light_count / 180, 1.0)
    return walk_score, light_score, light_count

for k, v in [('page','landing'), ('mlat', 32.7157), ('mlon', -117.1611), ('result', None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == 'landing':
    st.markdown('<div class="l-title">Urban Safety Perception</div>', unsafe_allow_html=True)
    st.markdown('<div class="l-sub">Predicting perceived safety across San Diego using machine learning</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.8, 1, 1.8])
    with c2:
        if st.button("Launch App", type="primary", use_container_width=True):
            st.session_state.page = 'app'
            st.rerun()

    st.markdown("""
    <div class="l-card">
        <div class="mrow"><span class="mlbl">Course</span>DSC 148 — Introduction to Data Mining</div>
        <div class="mrow"><span class="mlbl">By</span>Vanshika Somani</div>
        <div class="mrow"><span class="mlbl">Model</span>XGBoost &nbsp;|&nbsp; Accuracy: 99.9% &nbsp;|&nbsp; AUC-ROC: 1.000</div>
        <div class="mrow"><span class="mlbl">Dataset</span>7,872 San Diego grid points across 4 data sources</div>
        <div class="mrow"><span class="mlbl">GitHub</span>
            <a class="l-link" href="https://github.com/vanshika-s/urban-safety-perception" target="_blank">vanshika-s/urban-safety-perception</a>
        </div>
        <div class="mrow"><span class="mlbl">Report</span>
            <a class="l-link" href="https://placeholder-report-link.com" target="_blank">View Report (PDF)</a>
        </div>
    </div>
    <div class="l-about">
        This project predicts whether a San Diego location is perceived as safe or unsafe
        using environmental features: EPA walkability scores, streetlight density (56,000+ lights),
        and geographic context. Enter any address or click the map to get an instant safety
        prediction with feature-level explanations.
        <br><br>
        <b>Data Sources:</b>&nbsp; SDPD Calls for Service (2023) &nbsp;·&nbsp;
        EPA National Walkability Index &nbsp;·&nbsp;
        City of San Diego Streetlights &nbsp;·&nbsp;
        Census TIGER Block Groups
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# APP PAGE
# ══════════════════════════════════════════════════════════════════════════════
else:
    df = load_modeling_df()
    model = train_model(df)
    census_walk = load_census_walk()
    lights_proj = load_lights()

    left, right = st.columns([4, 6], gap="small")

    with left:
        st.markdown('<div class="panel-wrap">', unsafe_allow_html=True)

        hc1, hc2 = st.columns([3, 1])
        with hc1:
            st.markdown('<div class="panel-title">Urban Safety Perception</div>', unsafe_allow_html=True)
        with hc2:
            if st.button("Home"):
                st.session_state.page = 'landing'
                st.session_state.result = None
                st.rerun()

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        address = st.text_input("addr", label_visibility="collapsed",
                                placeholder="Search address or neighborhood...")

        geo_lat, geo_lon = None, None
        if address:
            geo_lat, geo_lon = geocode_address(address)
            if geo_lat and is_in_sd(geo_lat, geo_lon):
                st.session_state.mlat = geo_lat
                st.session_state.mlon = geo_lon
                st.markdown(f'<div style="font-size:0.82rem;color:#3a7abf;margin-bottom:4px;">Found: ({geo_lat:.4f}, {geo_lon:.4f})</div>', unsafe_allow_html=True)
            elif geo_lat:
                st.error("Location is outside San Diego.")
                geo_lat, geo_lon = None, None
            else:
                st.warning("Address not found.")

        c1, c2 = st.columns(2)
        with c1:
            lat_in = st.number_input("Lat", value=float(st.session_state.mlat),
                                     min_value=32.5, max_value=33.2, step=0.001, format="%.4f")
        with c2:
            lon_in = st.number_input("Lon", value=float(st.session_state.mlon),
                                     min_value=-117.5, max_value=-116.9, step=0.001, format="%.4f")

        final_lat = geo_lat if geo_lat else lat_in
        final_lon = geo_lon if geo_lon else lon_in

        if st.button("Predict Safety", type="primary", use_container_width=True):
            if not is_in_sd(final_lat, final_lon):
                st.error("Location is outside San Diego County.")
            else:
                with st.spinner("Computing..."):
                    walk_score, light_score, light_count = get_features(
                        final_lat, final_lon, census_walk, lights_proj)
                    X_pred = pd.DataFrame(
                        [[walk_score, light_score, final_lat, final_lon]],
                        columns=['walk_score', 'light_score', 'lat', 'lon'])
                    pred = int(model.predict(X_pred)[0])
                    prob = float(model.predict_proba(X_pred)[0][1])
                    st.session_state.result = dict(
                        pred=pred, prob=prob, walk=walk_score,
                        light=light_score, lights=light_count,
                        lat=final_lat, lon=final_lon)
                    st.session_state.mlat = final_lat
                    st.session_state.mlon = final_lon

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        if st.session_state.result:
            r = st.session_state.result
            if r['pred'] == 1:
                st.markdown(f'<div class="result-safe">SAFE &nbsp; Confidence: {r["prob"]:.1%}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-unsafe">UNSAFE &nbsp; Confidence: {1-r["prob"]:.1%}</div>', unsafe_allow_html=True)

            st.markdown("**Score Breakdown**")
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box"><div class="metric-val">{r['walk']:.3f}</div><div class="metric-lbl">Walkability</div></div>
                <div class="metric-box"><div class="metric-val">{r['light']:.3f}</div><div class="metric-lbl">Lighting</div></div>
                <div class="metric-box"><div class="metric-val">{r['lights']}</div><div class="metric-lbl">Streetlights</div></div>
                <div class="metric-box"><div class="metric-val">{r['prob']:.1%}</div><div class="metric-lbl">Safety Prob</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Feature Importance**")
            imp = pd.DataFrame({
                'Feature': ['Walkability', 'Lighting', 'Longitude', 'Latitude'],
                'SHAP Weight': [0.70, 0.21, 0.05, 0.04]
            })
            st.bar_chart(imp.set_index('Feature')['SHAP Weight'], height=150)
            st.markdown(f'<div style="font-size:0.78rem;color:#6c7a8d;">Location: ({r["lat"]:.4f}, {r["lon"]:.4f})</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        # Build map — don't rerun on click to preserve user zoom/pan
        m = folium.Map(
            location=[st.session_state.mlat, st.session_state.mlon],
            zoom_start=12,
            tiles='CartoDB positron'
        )

        # Selected location marker
        folium.CircleMarker(
            location=[st.session_state.mlat, st.session_state.mlon],
            radius=10,
            color='#3a7abf',
            fill=True,
            fill_color='#3a7abf',
            fill_opacity=0.9,
            popup=f"({st.session_state.mlat:.4f}, {st.session_state.mlon:.4f})"
        ).add_to(m)

        # Result marker
        if st.session_state.result:
            r = st.session_state.result
            color = '#28a745' if r['pred'] == 1 else '#dc3545'
            folium.CircleMarker(
                location=[r['lat'], r['lon']],
                radius=16,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.4,
                popup=f"{'SAFE' if r['pred']==1 else 'UNSAFE'} ({r['prob']:.1%})"
            ).add_to(m)

        map_data = st_folium(
            m,
            use_container_width=True,
            height=720,
            key="map",
            returned_objects=["last_clicked"]
        )

        # Read click — update coords without rerun so map keeps its position
        if map_data and map_data.get('last_clicked'):
            clat = round(map_data['last_clicked']['lat'], 4)
            clon = round(map_data['last_clicked']['lng'], 4)
            if abs(clat - round(st.session_state.mlat, 4)) > 0.0001 or \
               abs(clon - round(st.session_state.mlon, 4)) > 0.0001:
                st.session_state.mlat = clat
                st.session_state.mlon = clon
                st.session_state.result = None
                st.rerun()
