# JBG Logistics Route & Milk-Run Optimizer

An end-to-end applied machine learning and interactive spatial optimization web application engineered to plan transport routes, manage multi-stop milk-runs, predict segment risk, and estimate fuel expenditures across regional capitals and major highway corridors in Ghana.

🔗 **Live App:** [JBG Logistics Route Optimizer](https://jbg-logistics-rout-optimizer.streamlit.app/)

## 🚀 Key Features
* **Multi-Stop Milk-Run Optimization:** Automatically sequences delivery drop-off points using geographic angle loops and nearest-neighbor fallback heuristics.
* **Predictive Route Risk Classification:** Utilizes a trained machine learning model (`scikit-learn`) to evaluate path security, traffic levels, historical accidents, and weather conditions per segment.
* **Interactive Corridor Mapping:** Powered by `Folium` and `streamlit-folium` to plot route vectors, custom drop-off markers, and dynamic segment metric badges.
* **Logistics Analytics & Exporting:** Calculates real-time distance metrics, estimated travel times, and fuel costs (in GHS), with full support for exporting interactive HTML maps and printable driver manifests.
* **Containerized Deployment Ready:** Fully configured with a `Dockerfile` for seamless deployment across cloud environments.

## 🛠️ Tech Stack
* **Language:** Python
* **Web Framework:** Streamlit
* **Spatial Analysis & Mapping:** Folium, Streamlit-Folium, Math
* **Machine Learning & Modeling:** Scikit-Learn, Joblib, Pandas
* **Deployment & Ops:** Docker, Git / GitHub Actions


## 📂 Project Directory Structure
```text
JBG-Logistics-Optimizer/
│
├── models/                  # Trained machine learning artifacts & feature encoders
│   ├── route_recommendation_model.pkl
│   └── feature_encoders.pkl
│
├── src/                     # Core helper scripts and utilities
├── app.py                   # Main Streamlit application interface
├── Dockerfile               # Container configuration file
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

* ---
<p align="center">
  <small>Designed & Engineered by <b>Daniel Borffo Mensah</b> | Data Scientist & Machine Learning Engineer</small>
</p>
