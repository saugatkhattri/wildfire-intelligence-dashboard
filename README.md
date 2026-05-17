#  Wildfire Intelligence Dashboard

AI-assisted wildfire intelligence dashboard built using NASA FIRMS satellite detections, Flask, Mapbox GL JS, DBSCAN clustering, and rule-based / machine learning confidence scoring.

This project visualizes real-time satellite thermal anomalies across the United States and prioritizes potential wildfire detections using geospatial analytics and AI-assisted scoring techniques.

---

# 🛰️ Features

* Live NASA FIRMS VIIRS integration
* Real-time wildfire detection visualization
* DBSCAN spatial clustering of thermal anomalies
* Rule-based wildfire confidence scoring
* Optional machine learning wildfire probability scoring
* Interactive Mapbox Web GIS interface
* FRP (Fire Radiative Power) analysis
* Satellite metadata visualization
* Detection filtering and prioritization
* Historical SQLite detection persistence
* Modern operational dashboard UI

---

# ⚠️ Important Disclaimer

NASA FIRMS detects satellite thermal anomalies, NOT officially confirmed wildfire perimeters.

This dashboard is intended for:

* research
* geospatial analysis
* visualization
* situational awareness
* educational purposes

All detections should be validated using official wildfire agencies, emergency management authorities, and field observations.

---

#  System Architecture

## Backend

* Python
* Flask
* Flask-SocketIO
* SQLite
* Pandas
* NumPy
* Scikit-learn

## Frontend

* HTML
* CSS
* JavaScript
* Mapbox GL JS

## Data Source

* NASA FIRMS VIIRS Near Real-Time Data

---

# 🔬 Detection Intelligence Pipeline

The platform performs the following workflow:

1. Retrieve live VIIRS thermal anomaly detections from NASA FIRMS
2. Clean and engineer geospatial features
3. Apply DBSCAN spatial clustering
4. Compute rule-based confidence scoring using:

   * FRP
   * brightness
   * NASA confidence
   * cluster behavior
   * day/night analysis
5. Optionally apply machine learning probability scoring
6. Store detections in SQLite database
7. Visualize detections through interactive Web GIS dashboard

---

# 📊 Current Capabilities

* Satellite thermal anomaly monitoring
* AI-assisted detection prioritization
* Wildfire confidence visualization
* Cluster behavior analysis
* Operational wildfire situational awareness
* Real-time geospatial filtering

---

#  Planned Future Improvements

Future development ideas include:

* Terrain hillshade integration
* Wind particle visualization
* Smoke/weather overlays
* Time-series playback
* Fire spread direction estimation
* Historical fire behavior analysis
* Vegetation dryness integration
* GOES satellite support
* NIFC perimeter overlays
* Emergency response layers
* Evacuation route analysis
* Agency-grade operational tools

---

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/wildfire-intelligence-dashboard.git
cd wildfire-intelligence-dashboard
```

## Install Requirements

```bash
pip install -r requirements.txt
```

## Set Environment Variables

Create a `.env` file:

```env
MAPBOX_TOKEN=your_mapbox_token
NASA_API_KEY=your_nasa_api_key
```

## Run Application

```bash
python app.py
```

Application runs at:

```txt
http://127.0.0.1:5000
```

---

# 📁 Project Structure

```txt
wildfire-intelligence-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── database/
├── models/
├── screenshots/
└── .gitignore
```

---

# 🧭 Research & Development Goals

This project aims to explore:

* Geospatial intelligence systems
* Wildfire situational awareness
* Remote sensing analytics
* Real-time Web GIS applications
* AI-assisted environmental monitoring
* Emergency management visualization systems

---

# 🤝 Collaboration

I am interested in collaborating with:

* GIS developers
* wildfire researchers
* remote sensing analysts
* emergency management professionals
* geospatial AI researchers
* environmental monitoring organizations

If you are interested in extending this platform or collaborating on future development, feel free to connect.

---

# 📌 Technologies Used

* Python
* Flask
* SQLite
* Pandas
* NumPy
* Scikit-learn
* Mapbox GL JS
* NASA FIRMS API
* GeoJSON
* DBSCAN Clustering

---

# 📜 License

This project is released for educational and research purposes.

---

# Author

Saugat Khattri
GIS Graduate Student
Sam Houston State University

---
