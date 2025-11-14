import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from folium import PolyLine
import math
from pathlib import Path

# --------------------
# Helper functions
# --------------------
def haversine_km(lat1, lon1, lat2, lon2):
    # returns distance in km between two lat/lon
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def get_city_sample(df, start_city, end_city):
    # Return a representative sample row for the city pair if exists
    subset = df[(df['start_city']==start_city)&(df['end_city']==end_city)]
    if not subset.empty:
        # return median numeric + mode for categorical
        med = subset.median(numeric_only=True)
        mode_row = subset.mode().iloc[0].to_dict()
        out = med.to_dict()
        # add categorical fields from mode_row if present
        for c in ['weather','road_type','user_preference']:
            if c in subset.columns:
                out[c] = mode_row.get(c, None)
        # lat/lon take the first sample
        out['start_lat'] = float(subset.iloc[0]['start_lat'])
        out['start_lon'] = float(subset.iloc[0]['start_lon'])
        out['end_lat'] = float(subset.iloc[0]['end_lat'])
        out['end_lon'] = float(subset.iloc[0]['end_lon'])
        out['accidents'] = int(round(out.get('accidents', 0)))
        out['scenic_score'] = int(round(out.get('scenic_score', 0)))
        return out
    else:
        return None

def compute_leg_stats(lat1, lon1, lat2, lon2, approx_speed_kmph=60):
    dist = haversine_km(lat1, lon1, lat2, lon2)
    travel_time_min = (dist / max(approx_speed_kmph,5)) * 60
    fuel_cost = dist * 0.08 * 15.0  # litres per km * price per litre (approx)
    stats = {
        'distance_km': round(dist,2),
        'travel_time_min': round(travel_time_min,2),
        'fuel_cost_GHS': round(fuel_cost,2),
        'traffic_level': np.nan,
        'weather': None,
        'road_type': None,
        'accidents': 0,
        'safety_index': np.nan,
        'scenic_score': np.nan
    }
    return stats

# --------------------
# Streamlit page setup
# --------------------
st.set_page_config(page_title="Ghana Route Recommender", page_icon="🛣️", layout="wide")

st.title("🛣️ Ghana Route Recommendation — Multi-stop (Folium)")
st.markdown("Select start, optional stop, and destination. The app shows route map, per-leg breakdown and AI recommendation.")

# --------------------
# Paths & load resources
# --------------------
DATA_PATH = "/mnt/data/ghana_route_dataset_large.csv"
MODEL_PATHS = [
    "/mnt/data/route_recommender_best_model.joblib",
    "/mnt/data/route_model.pkl",
    "/mnt/data/models/route_recommender_best_model.joblib"
]

@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

@st.cache_resource(show_spinner=False)
def load_model(paths):
    for p in paths:
        try:
            model = joblib.load(p)
            return model, p
        except Exception:
            continue
    return None, None

model, model_path = load_model(MODEL_PATHS)

# --------------------
# Sidebar inputs
# --------------------
st.sidebar.header("Route selector")
cities = sorted(list(set(df['start_city'].unique()).union(set(df['end_city'].unique()))))
start = st.sidebar.selectbox("Start city", cities, index=cities.index('Accra') if 'Accra' in cities else 0)
stop_options = ["None"] + cities
stop = st.sidebar.selectbox("Optional stop (via)", stop_options, index=0)
end = st.sidebar.selectbox("Destination city", cities, index=cities.index('Kumasi') if 'Kumasi' in cities else 1)

if start == end:
    st.sidebar.error("Start and destination cannot be the same. Choose different cities.")

# --------------------
# Build itinerary (list of waypoints)
# --------------------
waypoints = [start]
if stop and stop != "None" and stop != start and stop != end:
    waypoints.append(stop)
waypoints.append(end)

# --------------------
# Gather data for each leg
# --------------------
legs = []
for i in range(len(waypoints)-1):
    s_city = waypoints[i]
    e_city = waypoints[i+1]
    sample = get_city_sample(df, s_city, e_city)
    if sample is not None:
        leg = {
            'start_city': s_city,
            'end_city': e_city,
            'start_lat': sample['start_lat'],
            'start_lon': sample['start_lon'],
            'end_lat': sample['end_lat'],
            'end_lon': sample['end_lon'],
            'distance_km': float(sample['distance_km']),
            'travel_time_min': float(sample['travel_time_min']),
            'traffic_level': int(sample.get('traffic_level', 1)) if not np.isnan(sample.get('traffic_level', np.nan)) else np.nan,
            'weather': sample.get('weather', None),
            'road_type': sample.get('road_type', None),
            'accidents': int(sample.get('accidents', 0)),
            'fuel_cost_GHS': float(sample.get('fuel_cost_GHS', np.nan)) if 'fuel_cost_GHS' in sample else np.nan,
            'safety_index': float(sample.get('safety_index', np.nan)) if 'safety_index' in sample else np.nan,
            'scenic_score': int(sample.get('scenic_score', 0)) if not np.isnan(sample.get('scenic_score', np.nan)) else 0
        }
    else:
        try:
            scoords = df[df['start_city']==s_city].iloc[0]
            ecoords = df[df['end_city']==e_city].iloc[0]
            s_lat, s_lon = float(scoords['start_lat']), float(scoords['start_lon'])
            e_lat, e_lon = float(ecoords['end_lat']), float(ecoords['end_lon'])
        except Exception:
            s_lat = df[df['start_city']==s_city]['start_lat'].mean() if not df[df['start_city']==s_city].empty else df['start_lat'].mean()
            s_lon = df[df['start_city']==s_city]['start_lon'].mean() if not df[df['start_city']==s_city].empty else df['start_lon'].mean()
            e_lat = df[df['end_city']==e_city]['end_lat'].mean() if not df[df['end_city']==e_city].empty else df['end_lat'].mean()
            e_lon = df[df['end_city']==e_city]['end_lon'].mean() if not df[df['end_city']==e_city].empty else df['end_lon'].mean()
        stats = compute_leg_stats(s_lat, s_lon, e_lat, e_lon)
        leg = {
            'start_city': s_city,
            'end_city': e_city,
            'start_lat': s_lat,
            'start_lon': s_lon,
            'end_lat': e_lat,
            'end_lon': e_lon,
            'distance_km': stats['distance_km'],
            'travel_time_min': stats['travel_time_min'],
            'traffic_level': np.nan,
            'weather': None,
            'road_type': None,
            'accidents': 0,
            'fuel_cost_GHS': stats['fuel_cost_GHS'],
            'safety_index': np.nan,
            'scenic_score': np.nan
        }
    legs.append(leg)

# --------------------
# Aggregate totals
# --------------------
total_distance = sum([l['distance_km'] for l in legs])
total_time = sum([l['travel_time_min'] for l in legs])
total_fuel = sum([l['fuel_cost_GHS'] for l in legs if not pd.isna(l['fuel_cost_GHS'])])
avg_safety = np.nanmean([l['safety_index'] for l in legs]) if any([not pd.isna(l['safety_index']) for l in legs]) else np.nan

# --------------------
# Left column: route summary and per-leg breakdown
# --------------------
left, right = st.columns([1,2])

with left:
    st.header("🧾 Route Summary")
    st.metric("Waypoints", " → ".join(waypoints))
    st.metric("Total distance (km)", f"{total_distance:.2f}")
    st.metric("Estimated total time (min)", f"{total_time:.1f}")
    st.metric("Estimated fuel cost (GHS)", f"{total_fuel:.2f}")
    if not np.isnan(avg_safety):
        st.metric("Average safety index", f"{avg_safety:.2f}/10")
    st.markdown("### Per-leg breakdown")
    for idx, l in enumerate(legs, 1):
        st.markdown(f"**Leg {idx}: {l['start_city']} → {l['end_city']}**")
        st.write({
            'distance_km': f"{l['distance_km']:.2f}",
            'travel_time_min': f"{l['travel_time_min']:.1f}",
            'fuel_cost_GHS': f\"{l['fuel_cost_GHS']:.2f}\" if not pd.isna(l['fuel_cost_GHS']) else 'N/A',
            'traffic_level': int(l['traffic_level']) if not pd.isna(l['traffic_level']) else 'N/A',
            'weather': l['weather'] if l['weather'] else 'N/A',
            'road_type': l['road_type'] if l['road_type'] else 'N/A',
            'accidents': int(l['accidents']) if 'accidents' in l else 0,
            'safety_index': f\"{l['safety_index']:.2f}\" if not pd.isna(l['safety_index']) else 'N/A'
        })

    st.markdown(\"---\")
    st.markdown(\"### AI Recommendation\")

    # Prepare features for model if available
    if model is not None:
        feat = {}
        feat['distance_km'] = total_distance
        feat['travel_time_min'] = total_time
        leg_traffic = [l['traffic_level'] for l in legs if not pd.isna(l['traffic_level'])]
        feat['traffic_level'] = int(np.nanmean(leg_traffic)) if leg_traffic else 1
        feat['accidents'] = int(sum([l.get('accidents',0) for l in legs]))
        feat['fuel_cost_GHS'] = float(total_fuel)
        feat['safety_index'] = float(avg_safety) if not np.isnan(avg_safety) else 7.0
        feat['scenic_score'] = int(np.nanmean([l['scenic_score'] for l in legs if not pd.isna(l['scenic_score'])]) if any([not pd.isna(l['scenic_score']) for l in legs]) else 5)
        try:
            X_pred = pd.DataFrame([feat])
            pred = model.predict(X_pred)[0]
            proba = model.predict_proba(X_pred)[0,1] if hasattr(model, "predict_proba") else None
            status = "✅ Recommended" if pred==1 else "❌ Not Recommended"
            st.success(f"Model verdict: **{status}**")
            if proba is not None:
                st.write(f"Model confidence (probability): {proba:.2f}")
            st.write("Model used from:", model.__class__.__name__, " (loaded from filesystem)")
        except Exception as e:
            st.error("Model prediction failed: " + str(e))
    else:
        score = 0
        score += max(0, (120 - total_time)/120) * 3
        score += ( (avg_safety if not np.isnan(avg_safety) else 7) / 10 ) * 3
        score += (1 / (1 + sum([l.get('accidents',0) for l in legs]))) * 1.5
        leg_traffic = [l['traffic_level'] for l in legs if not pd.isna(l['traffic_level'])]
        score += (3 - (int(np.nanmean(leg_traffic)) if leg_traffic else 1)) * 0.8
        status = "✅ Recommended" if score > 3.8 else "❌ Not Recommended"
        st.info("No trained model found. Showing rule-based recommendation:")
        st.write(f"Recommendation: **{status}**  — score = {score:.2f}")

with right:
    st.header("🗺️ Route Map")
    all_lats = [l['start_lat'] for l in legs] + [legs[-1]['end_lat']]
    all_lons = [l['start_lon'] for l in legs] + [legs[-1]['end_lon']]
    center_lat = float(np.mean(all_lats))
    center_lon = float(np.mean(all_lons))
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles='OpenStreetMap')

    colors = ['blue', 'green', 'purple', 'orange', 'cadetblue']
    for i, l in enumerate(legs):
        folium.Marker([l['start_lat'], l['start_lon']], popup=f"Start: {l['start_city']}", icon=folium.Icon(color='green')).add_to(m)
        if i == len(legs)-1:
            folium.Marker([l['end_lat'], l['end_lon']], popup=f"End: {l['end_city']}", icon=folium.Icon(color='red')).add_to(m)
        line_coords = [(l['start_lat'], l['start_lon']), (l['end_lat'], l['end_lon'])]
        PolyLine(locations=line_coords, color=colors[i % len(colors)], weight=5, opacity=0.8).add_to(m)
        mid_lat = (l['start_lat'] + l['end_lat']) / 2
        mid_lon = (l['start_lon'] + l['end_lon']) / 2
        folium.map.Marker([mid_lat, mid_lon], popup=(f"{l['start_city']}→{l['end_city']}: {l['distance_km']:.1f} km, {l['travel_time_min']:.1f} min"), icon=folium.DivIcon(html=f"<div style='font-size:12px;color:black'>{i+1}</div>")).add_to(m)

    st_folium(m, width=900, height=600)

st.markdown("---")
st.caption("This app uses a Ghana-specific synthetic dataset. Developed by Borffo.")