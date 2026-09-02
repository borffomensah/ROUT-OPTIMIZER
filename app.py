"""
=============================================================================
Author: Daniel Borffo Mensah
Project: JBG Logistics Route Optimizer & ML Risk Predictor
Description: Multi-stop logistics corridor planner across Ghanaian highways.
=============================================================================
"""

import math
import folium
import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Load trained artifacts
model = joblib.load("models/route_recommendation_model.pkl")
encoders = joblib.load("models/feature_encoders.pkl")

st.title("JBG Logistics Route Optimizer")
st.write(
    "Plan optimal transportation routes, estimate fuel costs, manage multi-stop "
    "milk-runs, and visualize corridor paths across all regional capitals and connecting highway towns in Ghana."
)

# Comprehensive dictionary containing all regional capitals and major highway towns
hub_coords = {
    # --- All 16 Regional Capitals ---
    "Accra": [5.5560, -0.1969],
    "Tema": [5.6667, -0.0167],
    "Kumasi": [6.7000, -1.6250],
    "Tamale": [9.4075, -0.8533],
    "Takoradi": [4.8800, -1.7600],
    "Cape Coast": [5.1000, -1.2500],
    "Sunyani": [7.3333, -2.3333],
    "Koforidua": [6.0941, -0.2591],
    "Ho": [6.6114, 0.4703],
    "Wa": [10.0667, -2.5000],
    "Bolgatanga": [10.7856, -0.8514],
    "Damongo": [9.0830, -1.8188],
    "Nalerigu": [10.5273, -0.3698],
    "Techiman": [7.5905, -1.9395],
    "Goaso": [6.8000, -2.5167],
    "Dambai": [8.0689, 0.1792],
    "Sefwi Wiawso": [6.2158, -2.4850],

    # --- N1 Highway Corridor (Coastal: Elubo to Aflao) ---
    "Elubo": [5.2833, -2.7667],
    "Axim": [4.8667, -2.2417],
    "Agona Nkwanta": [4.8333, -1.9667],
    "Sekondi": [4.9340, -1.7130],
    "Shama": [5.0087, -1.6301],
    "Elmina": [5.0847, -1.3508],
    "Saltpond": [5.2108, -1.0547],
    "Mankessim": [5.2414, -1.0107],
    "Apam": [5.2941, -0.7390],
    "Winneba": [5.3511, -0.6231],
    "Kasoa": [5.5345, -0.4168],
    "Swedru": [5.5355, -0.6974],
    "Sege": [5.7833, 0.4333],
    "Aflao": [6.1198, 1.1901],

    # --- N2 Highway Corridor (Tema to Kulungugu / Bawku) ---
    "Kpong": [6.1481, 0.0575],
    "Adomi": [6.2575, 0.0469],
    "Asikuma": [6.6881, 0.1878],
    "Hohoe": [7.1514, 0.4737],
    "Jasikan": [7.3822, 0.4783],
    "Kadjebi": [7.5772, 0.4678],
    "Nkwanta": [8.2550, 0.7511],
    "Bimbila": [8.8631, 0.0592],
    "Yendi": [9.4447, -0.0099],
    "Bawku": [11.0616, -0.2417],
    "Kulungugu": [11.0800, -0.3200],

    # --- N3 & N4 Highway Corridors (Eastern Links) ---
    "Somanya": [6.0967, -0.0133],
    "Adenta": [5.7170, -0.1650],
    "Mamfe": [5.9575, -0.1264],
    "Bunso": [6.2917, -0.4650],

    # --- N6 Highway Corridor (Accra – Kumasi) ---
    "Nsawam": [5.8080, -0.3537],
    "Suhum": [6.0400, -0.4500],
    "Apedwa": [6.1833, -0.4167],
    "Anyinam": [6.4883, -0.5694],
    "Nkawkaw": [6.5500, -0.7700],
    "Juaso": [6.5167, -1.0833],
    "Konongo": [6.6167, -1.2167],
    "Ejisu": [6.7214, -1.4722],

    # --- N7, N12 & Western North Corridors ---
    "Larabanga": [9.2333, -2.7167],
    "Fufulsu": [8.9333, -1.3333],
    "Bamboi": [8.1333, -2.0500],
    "Enchi": [5.8286, -2.8153],
    "Lawra": [10.6558, -2.8906],
    "Hamile": [10.8000, -2.8167],

    # --- N8 & Cross-Links (Central / Ashanti Trunk) ---
    "Assin Fosu": [5.7114, -1.4061],
    "Twifo Praso": [5.6128, -1.5478],
    "Dunkwa": [5.9667, -1.7833],
    "Fomena": [6.1833, -1.6000],
    "Bekwai": [6.4528, -1.5819],
    "Obuasi": [6.2023, -1.6680],

    # --- N10 & Northern Highways ---
    "Kintampo": [8.0563, -1.7306],
    "Buipe": [8.3333, -1.2500],
    "Yapei": [9.1333, -1.1500],
    "Zebilla": [10.8856, -0.5281],

    # --- Additional Connecting Nodes & Regional Towns ---
    "Dodowa": [5.8756, -0.1086],
    "Juapong": [6.1167, 0.1167],
    "Peki": [6.6136, 0.2869],
    "Sogakope": [6.0022, 0.5925],
    "Akatsi": [6.1242, 0.8061],
    "Berekum": [7.4532, -2.5838],
    "Dormaa-Ahenkro": [7.2833, -2.8833],
    "Duayaw Nkwanta": [7.3117, -2.1317],
    "Bole": [9.0333, -2.4833],
    "Sawla": [9.2708, -2.4183],
    "Navrongo": [10.8987, -1.0920],
    "Tumu": [10.3392, -1.9789],
}

def optimize_stop_order(start, stops, end, coords_dict, method="angle"):
    """Sorts stops logically based on chosen strategy (Angle Loop, Reverse Angle, or Nearest Neighbor)."""
    if not stops:
        return []
    
    MAX_LEG_JUMP_DEG = 4.2 
    def get_dist_deg(c1, c2):
        p1 = coords_dict.get(c1, [0, 0])
        p2 = coords_dict.get(c2, [0, 0])
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    valid_stops = [s for s in stops if get_dist_deg(start, s) <= MAX_LEG_JUMP_DEG or start == end]

    if method == "angle" and start == end and len(valid_stops) > 2:
        center_lat, center_lon = 6.7, -1.5
        return sorted(valid_stops, key=lambda c: math.atan2(coords_dict[c][0] - center_lat, coords_dict[c][1] - center_lon))
    elif method == "reverse_angle" and start == end and len(valid_stops) > 2:
        center_lat, center_lon = 6.7, -1.5
        return sorted(valid_stops, key=lambda c: math.atan2(coords_dict[c][0] - center_lat, coords_dict[c][1] - center_lon), reverse=True)
    
    # Nearest-Neighbor fallback
    unvisited = list(valid_stops)
    current = start
    ordered_stops = []
    while unvisited:
        next_stop = min(unvisited, key=lambda city: get_dist_deg(current, city))
        ordered_stops.append(next_stop)
        current = next_stop
        unvisited.remove(next_stop)
        
    return ordered_stops

# Initialize Session State safely
if "optimized" not in st.session_state:
    st.session_state.optimized = False
    st.session_state.prediction = None
    st.session_state.proba_val = 0.0
    st.session_state.corridor_choice = None
    st.session_state.legs = []
    st.session_state.total_distance = 0.0
    st.session_state.total_time = 0.0
    st.session_state.path_coords = []
    st.session_state.full_path_names = []
    st.session_state.strategy_comparison = []
    st.session_state.best_strategy_name = ""

# Sidebar inputs for user selection
st.sidebar.header("Route & Milk-Run Parameters")

available_cities = list(hub_coords.keys())
start_city = st.sidebar.selectbox("Start City / Port", available_cities, index=1)  # Default Tema
end_city = st.sidebar.selectbox("Final Destination", available_cities, index=5)    # Default Cape Coast

# Intermediate Drop-off Stops Selection
intermediate_stops = st.sidebar.multiselect(
    "Select Intermediate Drop-Off Stops (Milk Run Corridor)",
    [c for c in available_cities if c not in [start_city, end_city]],
    default=["Kasoa", "Winneba", "Apam", "Mankessim"] if start_city == "Tema" and end_city == "Cape Coast" else []
)

route_strategy = st.sidebar.selectbox(
    "Select Route Strategy / Corridor Type",
    [
        "Primary National Highway (Fastest / Paved)",
        "Secondary Regional Bypass (Moderate)",
        "Local / Feeder Track (Shorter distance, higher risk)",
    ],
)

traffic_level = st.sidebar.selectbox(
    "Traffic Level",
    [1, 2, 3],
    format_func=lambda x: ["Light", "Moderate", "Heavy"][x - 1],
)
weather = st.sidebar.selectbox(
    "Weather Condition", encoders["weather"].classes_
)
road_type = st.sidebar.selectbox("Road Type", encoders["road_type"].classes_)
accidents = st.sidebar.slider("Historical Accidents Count (Per Leg)", 0, 5, 0)

if st.sidebar.button("Optimize & Compare Routes"):
    if not intermediate_stops:
        st.sidebar.error("Please select at least one intermediate drop-off stop for multi-stop optimization.")
    else:
        # Define candidate sorting strategies to evaluate efficiency
        candidate_strategies = {
            "Geographic Angle Loop (Standard)": optimize_stop_order(start_city, intermediate_stops, end_city, hub_coords, method="angle"),
            "Geographic Angle Loop (Reversed)": optimize_stop_order(start_city, intermediate_stops, end_city, hub_coords, method="reverse_angle"),
            "Nearest-Neighbor Circuit": optimize_stop_order(start_city, intermediate_stops, end_city, hub_coords, method="nn")
        }

        comparison_records = []
        strategy_store = {}

        for strat_name, optimized_intermediates in candidate_strategies.items():
            full_path = [start_city] + optimized_intermediates + [end_city]
            leg_records = []
            total_dist = 0.0
            total_time = 0.0
            path_latlons = []
            
            overall_recommendation = 1
            min_proba = 1.0

            for i in range(len(full_path) - 1):
                leg_start = full_path[i]
                leg_end = full_path[i+1]
                
                p1 = hub_coords.get(leg_start, [5.6, -0.1])
                p2 = hub_coords.get(leg_end, [5.6, -0.1])
                
                approx_dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5 * 111.0 * 1.3
                approx_dist = max(approx_dist, 8.0)
                
                if "Primary National Highway" in route_strategy:
                    leg_dist = approx_dist
                    leg_time = (leg_dist / 65.0) * 60.0
                    road_val = "Main"
                elif "Secondary Regional Bypass" in route_strategy:
                    leg_dist = approx_dist * 1.08
                    leg_time = (leg_dist / 55.0) * 60.0
                    road_val = "Main"
                else:
                    leg_dist = approx_dist * 0.95
                    leg_time = (leg_dist / 40.0) * 60.0
                    road_val = "Local"

                input_df = pd.DataFrame(
                    [[
                        leg_start,
                        leg_end,
                        leg_dist,
                        leg_time,
                        traffic_level,
                        weather,
                        road_val,
                        accidents,
                        leg_dist * 2.1,
                        8.2,
                        5,
                        65.0,
                    ]],
                    columns=[
                        "start_city",
                        "end_city",
                        "distance_km",
                        "travel_time_min",
                        "traffic_level",
                        "weather",
                        "road_type",
                        "accidents",
                        "fuel_cost",
                        "safety_index",
                        "scenic_score",
                        "avg_speed_kmph",
                    ],
                )

                for col, enc_key in [("start_city", "start_city"), ("end_city", "end_city"), ("weather", "weather")]:
                    if input_df[col].iloc[0] in encoders[enc_key].classes_:
                        input_df[col] = encoders[enc_key].transform(input_df[col])
                    else:
                        input_df[col] = 0

                if road_val in encoders["road_type"].classes_:
                    input_df["road_type"] = encoders["road_type"].transform([road_val])[0]
                else:
                    input_df["road_type"] = 0

                if leg_dist > 450.0 or leg_time > 360.0:
                    pred = 0
                    conf = 0.99
                    overall_recommendation = 0
                    if conf < min_proba:
                        min_proba = conf
                else:
                    pred = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0]
                    conf = proba[pred]

                    if pred == 0:
                        overall_recommendation = 0
                    if conf < min_proba:
                        min_proba = conf

                total_dist += leg_dist
                total_time += leg_time
                
                leg_records.append({
                    "Leg": f"{leg_start} ➔ {leg_end}",
                    "Distance (km)": round(leg_dist, 1),
                    "Time (hrs)": round(leg_time / 60.0, 1),
                    "Fuel Cost (GHS)": round(leg_dist * 2.1, 2),
                    "Status": "Recommended" if pred == 1 else "Caution / Risk"
                })

                path_latlons.append(hub_coords.get(leg_start))
            
            path_latlons.append(hub_coords.get(full_path[-1]))

            strategy_store[strat_name] = {
                "prediction": overall_recommendation,
                "proba_val": min_proba,
                "legs": leg_records,
                "total_distance": total_dist,
                "total_time": total_time,
                "path_coords": path_latlons,
                "full_path_names": full_path
            }

            comparison_records.append({
                "Strategy": strat_name,
                "First Stop": full_path[1],
                "Total Distance (km)": round(total_dist, 1),
                "Total Time (hrs)": round(total_time / 60.0, 1),
                "Fuel Cost (GHS)": round(total_dist * 2.1, 2)
            })

        # Automatically determine the best strategy based on minimal total distance/fuel efficiency
        best_strat_info = min(comparison_records, key=lambda x: x["Total Distance (km)"])
        best_name = best_strat_info["Strategy"]

        st.session_state.optimized = True
        st.session_state.strategy_comparison = comparison_records
        st.session_state.best_strategy_name = best_name
        
        # Load best strategy details into active session state for mapping and display
        best_data = strategy_store[best_name]
        st.session_state.prediction = best_data["prediction"]
        st.session_state.proba_val = best_data["proba_val"]
        st.session_state.corridor_choice = route_strategy
        st.session_state.legs = best_data["legs"]
        st.session_state.total_distance = best_data["total_distance"]
        st.session_state.total_time = best_data["total_time"]
        st.session_state.path_coords = best_data["path_coords"]
        st.session_state.full_path_names = best_data["full_path_names"]

# Display results if optimization has been run
if st.session_state.optimized:
    st.subheader("📊 Routing Strategy Efficiency Comparison")
    comp_df = pd.DataFrame(st.session_state.strategy_comparison)
    st.dataframe(comp_df, use_container_width=True)

    best_strat = st.session_state.best_strategy_name
    first_stop_name = st.session_state.full_path_names[1] if len(st.session_state.full_path_names) > 1 else "Destination"
    
    st.success(
        f"🏆 **Optimal Conclusion:** **{best_strat}** is recommended to start first. "
        f"It begins by heading directly to **{first_stop_name}**, optimizing fuel efficiency at "
        f"**GHS {st.session_state.total_distance * 2.1:.2f}** and total distance of **{st.session_state.total_distance:.1f} km**."
    )

    st.subheader("Selected Optimal Route Manifest")
    if st.session_state.prediction == 1:
        st.success(
            f"✅ **Complete Multi-Stop Route Recommended via {st.session_state.corridor_choice}!** "
            f"(Min Leg Confidence: {st.session_state.proba_val*100:.1f}%)"
        )
    else:
        st.warning(
            f"⚠️ **Route Contains High-Risk Segments** due to excessive distance, driver fatigue risk, or adverse conditions."
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Distance", f"{st.session_state.total_distance:.1f} km")
    col2.metric("Total Est. Time", f"{st.session_state.total_time / 60:.1f} hours")
    col3.metric("Total Est. Fuel Cost", f"GHS {st.session_state.total_distance * 2.1:.2f}")

    st.markdown("### 📋 Segment-by-Segment Manifest")
    legs_df = pd.DataFrame(st.session_state.legs)
    st.dataframe(legs_df, use_container_width=True)

    # Interactive Map Creation
    map_center = (
        st.session_state.path_coords[0]
        if st.session_state.path_coords
        else [7.9465, -1.0232]
    )

    ghana_map = folium.Map(
        location=map_center,
        zoom_start=8,
        min_zoom=7,
        max_zoom=16,
        max_bounds=True,
        min_lat=4.5,
        max_lat=11.5,
        min_lon=-3.5,
        max_lon=1.5,
        control_scale=True,
    )

    # Add markers with sequential drop-off numbering
    full_path = st.session_state.full_path_names
    dropoff_counter = 1
    
    for idx, city_name in enumerate(full_path):
        coords = hub_coords.get(city_name, [5.6, -0.1])
        if idx == 0:
            icon_color = "blue"
            icon_type = "play"
            role_label = "Origin"
            display_label = f"📍 {city_name} (Origin)"
        elif idx == len(full_path) - 1:
            icon_color = "red"
            icon_type = "flag"
            role_label = "Destination"
            display_label = f"📍 {city_name} (Destination)"
        else:
            icon_color = "green"
            icon_type = "stop"
            role_label = f"Drop-off #{dropoff_counter}"
            display_label = f"📍 {dropoff_counter}. {city_name} <span style='font-size: 8px; color: #555;'>(Drop-off #{dropoff_counter})</span>"
            dropoff_counter += 1

        folium.Marker(
            coords,
            popup=f"<b>{city_name}</b><br>Role: {role_label}",
            icon=folium.Icon(color=icon_color, icon=icon_type, prefix="fa"),
        ).add_to(ghana_map)

        lat_offset = -0.035 if idx % 2 == 0 else 0.035
        
        town_label_html = f"""
        <div style="font-size: 10px; font-weight: bold; color: #1a237e; background: rgba(255,255,255,0.95); padding: 2px 4px; border-radius: 3px; border: 1px solid #3f51b5; white-space: nowrap; box-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
            {display_label}
        </div>
        """
        folium.Marker(
            [coords[0] + lat_offset, coords[1]],
            icon=folium.DivIcon(html=town_label_html)
        ).add_to(ghana_map)

    # Draw corridor paths and stagger metric badges
    for i in range(len(full_path) - 1):
        p1 = hub_coords.get(full_path[i])
        p2 = hub_coords.get(full_path[i+1])
        
        folium.PolyLine(
            [p1, p2],
            color="green" if st.session_state.prediction == 1 else "orange",
            weight=4,
            opacity=0.8
        ).add_to(ghana_map)

        mid_lat = (p1[0] + p2[0]) / 2
        mid_lon = (p1[1] + p2[1]) / 2
        
        metric_lat_offset = 0.045 if i % 2 == 0 else -0.045
        
        leg_data = st.session_state.legs[i]
        metric_html = f"""
        <div style="background-color: #ffffff; padding: 4px 6px; border: 1px solid #2e7d32; font-size: 9px; border-radius: 4px; box-shadow: 1px 1px 4px rgba(0,0,0,0.3); white-space: nowrap;">
            <b>{leg_data['Leg']}</b><br>
            📏 {leg_data['Distance (km)']}km | ⏱️ {leg_data['Time (hrs)']}h<br>
            ⛽ GHS {leg_data['Fuel Cost (GHS)']}
        </div>
        """
        folium.Marker(
            [mid_lat + metric_lat_offset, mid_lon],
            icon=folium.DivIcon(html=metric_html)
        ).add_to(ghana_map)

    # Download Options for Map & Driver Report
    st.markdown("### 📥 Export Route & Map Reports")
    dl_col1, dl_col2 = st.columns(2)

    map_html_data = ghana_map._repr_html_().encode("utf-8")
    dl_col1.download_button(
        label="Download Route Map (HTML)",
        data=map_html_data,
        file_name="ghana_route_map.html",
        mime="text/html",
        help="Download the interactive map file for management review and browser printing."
    )

    report_text = f"""==================================================
GHANA SMART ROUTE & LOGISTICS - DRIVER MANIFEST
==================================================
Winning Strategy: {st.session_state.best_strategy_name}
Route Strategy: {st.session_state.corridor_choice}
Total Distance: {st.session_state.total_distance:.1f} km
Total Estimated Time: {st.session_state.total_time / 60:.1f} hours
Total Estimated Fuel Cost: GHS {st.session_state.total_distance * 2.1:.2f}

--------------------------------------------------
SEGMENT DETAILS:
--------------------------------------------------
"""
    for leg in st.session_state.legs:
        report_text += f"• Leg: {leg['Leg']}\n"
        report_text += f"  - Distance: {leg['Distance (km)']} km\n"
        report_text += f"  - Time: {leg['Time (hrs)']} hrs\n"
        report_text += f"  - Fuel Cost: GHS {leg['Fuel Cost (GHS)']}\n"
        report_text += f"  - Status: {leg['Status']}\n\n"

    dl_col2.download_button(
        label="Download Printable Report (TXT)",
        data=report_text,
        file_name="driver_route_report.txt",
        mime="text/plain",
        help="Download text summary for printing and driver briefing."
    )

    st.subheader("Interactive Multi-Stop Corridor Mapping")
    st_folium(ghana_map, width=750, height=550)


st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ System & Author Info"):
    st.caption("**System:** JBG Logistics v1.0")
    st.caption("**Engineer:** Daniel Borffo Mensah")
    st.caption("**Tech Stack:** Python, Streamlit, Folium, Scikit-Learn")