"""
Wildfire Intelligence Backend
─────────────────────────────
AI-assisted wildfire intelligence dashboard using NASA FIRMS VIIRS data,
DBSCAN clustering, rule-based confidence scoring, optional ML inference,
SQLite persistence, and Flask GeoJSON APIs.

Important:
NASA FIRMS detects satellite thermal anomalies, not officially confirmed
wildfire perimeters. This app is for research, visualization, and situational
awareness only.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO
from sklearn.cluster import DBSCAN

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────────────────────

load_dotenv()

MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")
NASA_API_KEY = os.getenv("NASA_API_KEY")

if not MAPBOX_TOKEN:
    raise ValueError("Missing MAPBOX_TOKEN. Add it to your .env file.")

if not NASA_API_KEY:
    raise ValueError("Missing NASA_API_KEY. Add it to your .env file.")


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

USA_BBOX = "-125,24,-66,50"
FIRMS_DAYS = 1

DBSCAN_EPS_KM = 0.5
DBSCAN_EPS_RAD = DBSCAN_EPS_KM / 6371.0
DBSCAN_MIN_SAMPLES = 2

MODEL_PATH = "models/wildfire_rf_model.joblib"

DB_FOLDER = "database"
DB_PATH = os.path.join(DB_FOLDER, "fire_history.db")

NASA_CONF_MAP = {
    "low": 0.2,
    "nominal": 0.6,
    "high": 1.0,
    "n": 0.0,
}

ML_FEATURE_COLS = [
    "frp",
    "brightness",
    "confidence_numeric",
    "cluster_size",
    "scan",
    "track",
    "latitude",
    "longitude",
    "hour",
]

_MODEL: Optional[object] = None


# ─────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────

def initialize_database() -> None:
    os.makedirs(DB_FOLDER, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fires (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fire_uid TEXT UNIQUE NOT NULL,
                latitude REAL,
                longitude REAL,
                brightness REAL,
                confidence TEXT,
                frp REAL,
                acq_date TEXT,
                acq_time TEXT,
                satellite TEXT,
                instrument TEXT,
                scan REAL,
                track REAL,
                cluster_id INTEGER,
                cluster_size INTEGER,
                rule_score REAL,
                rule_label TEXT,
                ml_probability REAL,
                final_label TEXT,
                fire_status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    log.info("Database ready: %s", DB_PATH)


# ─────────────────────────────────────────────────────────────
# NASA FIRMS DATA
# ─────────────────────────────────────────────────────────────

def get_firms_data() -> pd.DataFrame:
    url = (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{NASA_API_KEY}/VIIRS_SNPP_NRT/{USA_BBOX}/{FIRMS_DAYS}"
    )

    log.info("Fetching NASA FIRMS VIIRS data...")

    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"NASA FIRMS network error: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"NASA FIRMS returned HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    if "latitude" not in response.text.lower():
        raise RuntimeError(
            "NASA FIRMS did not return expected CSV data. "
            f"Preview: {response.text[:300]}"
        )

    df = pd.read_csv(StringIO(response.text))
    df = df.dropna(subset=["latitude", "longitude"])

    log.info("Received %d raw satellite detections", len(df))
    return df


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def parse_confidence(value: object) -> float:
    text = str(value).strip().lower()

    if text in NASA_CONF_MAP:
        return NASA_CONF_MAP[text]

    try:
        return float(text) / 100.0
    except ValueError:
        return 0.5


def parse_hour(value: object) -> float:
    try:
        text = str(int(value)).zfill(4)
        return float(text[:2])
    except Exception:
        return -1.0


def make_fire_uid(lat: float, lon: float, acq_date: str, acq_time: str) -> str:
    raw = f"{round(lat, 4)}|{round(lon, 4)}|{acq_date}|{acq_time}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "bright_ti4" in df.columns:
        df["brightness"] = pd.to_numeric(df["bright_ti4"], errors="coerce").fillna(0)
    else:
        df["brightness"] = pd.to_numeric(df.get("brightness", 0), errors="coerce").fillna(0)

    df["frp"] = pd.to_numeric(df.get("frp", 0), errors="coerce").fillna(0)
    df["scan"] = pd.to_numeric(df.get("scan", 0), errors="coerce").fillna(0)
    df["track"] = pd.to_numeric(df.get("track", 0), errors="coerce").fillna(0)

    df["confidence_raw"] = df.get("confidence", "nominal")
    df["confidence_numeric"] = df["confidence_raw"].apply(parse_confidence)

    df["hour"] = df.get("acq_time", pd.Series([-1] * len(df))).apply(parse_hour)
    df["daynight"] = df.get("daynight", "U")

    df["acq_date"] = df.get("acq_date", "").astype(str)
    df["acq_time"] = df.get("acq_time", "").astype(str)
    df["satellite"] = df.get("satellite", "").astype(str)
    df["instrument"] = df.get("instrument", "").astype(str)

    df["fire_uid"] = df.apply(
        lambda row: make_fire_uid(
            row["latitude"],
            row["longitude"],
            row["acq_date"],
            row["acq_time"],
        ),
        axis=1,
    )

    return df


# ─────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────

def cluster_detections(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if df.empty:
        df["cluster_id"] = -1
        df["cluster_size"] = 1
        return df

    coords_rad = np.radians(df[["latitude", "longitude"]].values)

    labels = DBSCAN(
        eps=DBSCAN_EPS_RAD,
        min_samples=DBSCAN_MIN_SAMPLES,
        algorithm="ball_tree",
        metric="haversine",
    ).fit_predict(coords_rad)

    df["cluster_id"] = labels

    cluster_sizes = (
        df[df["cluster_id"] >= 0]["cluster_id"]
        .value_counts()
        .to_dict()
    )

    df["cluster_size"] = df["cluster_id"].apply(
        lambda cid: cluster_sizes.get(cid, 1) if cid >= 0 else 1
    )

    log.info(
        "DBSCAN complete: %d clusters, %d noise points",
        len(set(labels) - {-1}),
        int((labels == -1).sum()),
    )

    return df


# ─────────────────────────────────────────────────────────────
# RULE-BASED SCORING
# ─────────────────────────────────────────────────────────────

def rule_based_score(row: pd.Series) -> float:
    score = 0.0

    frp = float(row.get("frp", 0) or 0)
    brightness = float(row.get("brightness", 0) or 0)
    confidence = float(row.get("confidence_numeric", 0.5) or 0.5)
    cluster_size = int(row.get("cluster_size", 1) or 1)
    daynight = str(row.get("daynight", "U")).upper()

    if frp >= 100:
        score += 0.30
    elif frp >= 50:
        score += 0.22
    elif frp >= 20:
        score += 0.14
    elif frp >= 5:
        score += 0.08
    else:
        score += 0.02

    if brightness >= 370:
        score += 0.25
    elif brightness >= 340:
        score += 0.18
    elif brightness >= 320:
        score += 0.10
    else:
        score += 0.02

    score += 0.20 * confidence

    if cluster_size >= 10:
        score += 0.15
    elif cluster_size >= 5:
        score += 0.10
    elif cluster_size >= 2:
        score += 0.05

    if daynight == "D":
        score += 0.05
    elif daynight == "N":
        score += 0.08

    return min(round(score, 4), 1.0)


def rule_label(score: float) -> str:
    if score >= 0.60:
        return "high_confidence_wildfire"
    if score >= 0.35:
        return "possible_fire"
    return "low_confidence_or_false_positive"


def apply_rule_scoring(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rule_score"] = df.apply(rule_based_score, axis=1)
    df["rule_label"] = df["rule_score"].apply(rule_label)
    return df


# ─────────────────────────────────────────────────────────────
# OPTIONAL ML SCORING
# ─────────────────────────────────────────────────────────────

def load_model() -> Optional[object]:
    global _MODEL

    if not JOBLIB_AVAILABLE:
        return None

    if _MODEL is not None:
        return _MODEL

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        _MODEL = joblib.load(MODEL_PATH)
        log.info("ML model loaded: %s", MODEL_PATH)
    except Exception as exc:
        log.warning("Could not load ML model. Using rule-based scoring. Error: %s", exc)
        _MODEL = None

    return _MODEL


def apply_ml_scoring(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    model = load_model()

    if model is None:
        df["ml_probability"] = df["rule_score"]
        df["final_label"] = df["rule_label"]
        return df

    try:
        x = df[ML_FEATURE_COLS].fillna(0).values.astype(float)
        probabilities = model.predict_proba(x)[:, 1]

        df["ml_probability"] = np.round(probabilities, 4)
        df["final_label"] = df["ml_probability"].apply(rule_label)

        log.info("ML inference completed for %d detections", len(df))

    except Exception as exc:
        log.warning("ML inference failed. Using rule-based scoring. Error: %s", exc)
        df["ml_probability"] = df["rule_score"]
        df["final_label"] = df["rule_label"]

    return df


# ─────────────────────────────────────────────────────────────
# DATABASE PERSISTENCE
# ─────────────────────────────────────────────────────────────

def get_existing_uids() -> set[str]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT fire_uid FROM fires").fetchall()
        return {row[0] for row in rows}
    except Exception:
        return set()


def persist_fires(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    existing_uids = get_existing_uids()

    df["fire_status"] = df["fire_uid"].apply(
        lambda uid: "existing" if uid in existing_uids else "new"
    )

    new_rows = df[df["fire_status"] == "new"]

    log.info(
        "%d new detections, %d existing detections",
        len(new_rows),
        len(df) - len(new_rows),
    )

    if new_rows.empty:
        return df

    records = [
        (
            row["fire_uid"],
            row["latitude"],
            row["longitude"],
            row["brightness"],
            row["confidence_raw"],
            row["frp"],
            row["acq_date"],
            row["acq_time"],
            row["satellite"],
            row["instrument"],
            row["scan"],
            row["track"],
            int(row["cluster_id"]),
            int(row["cluster_size"]),
            float(row["rule_score"]),
            row["rule_label"],
            float(row["ml_probability"]),
            row["final_label"],
            row["fire_status"],
        )
        for _, row in new_rows.iterrows()
    ]

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO fires (
                    fire_uid,
                    latitude,
                    longitude,
                    brightness,
                    confidence,
                    frp,
                    acq_date,
                    acq_time,
                    satellite,
                    instrument,
                    scan,
                    track,
                    cluster_id,
                    cluster_size,
                    rule_score,
                    rule_label,
                    ml_probability,
                    final_label,
                    fire_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()

    except sqlite3.Error as exc:
        log.error("Database write error: %s", exc)

    return df


# ─────────────────────────────────────────────────────────────
# GEOJSON
# ─────────────────────────────────────────────────────────────

def build_geojson(df: pd.DataFrame) -> dict:
    features = []

    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    float(row["longitude"]),
                    float(row["latitude"]),
                ],
            },
            "properties": {
                "fire_uid": row["fire_uid"],
                "status": row["fire_status"],
                "frp": float(row["frp"]),
                "brightness": float(row["brightness"]),
                "nasa_confidence": row["confidence_raw"],
                "cluster_id": int(row["cluster_id"]),
                "cluster_size": int(row["cluster_size"]),
                "rule_score": float(row["rule_score"]),
                "rule_label": row["rule_label"],
                "ml_probability": float(row["ml_probability"]),
                "final_label": row["final_label"],
                "date": row["acq_date"],
                "time": row["acq_time"],
                "satellite": row["satellite"],
                "instrument": row["instrument"],
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline() -> dict:
    raw_df = get_firms_data()
    feature_df = engineer_features(raw_df)
    clustered_df = cluster_detections(feature_df)
    scored_df = apply_rule_scoring(clustered_df)
    ml_df = apply_ml_scoring(scored_df)
    final_df = persist_fires(ml_df)

    return build_geojson(final_df)


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", mapbox_token=MAPBOX_TOKEN)


@app.route("/api/fires")
def fires():
    try:
        return jsonify(run_pipeline())
    except Exception as exc:
        log.exception("Error in /api/fires")
        return jsonify({
            "type": "FeatureCollection",
            "features": [],
            "error": str(exc),
        }), 500


@app.route("/api/summary")
def summary():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query("SELECT * FROM fires", conn)
    except Exception as exc:
        log.error("Database read error: %s", exc)
        return jsonify({"error": str(exc)}), 500

    if df.empty:
        return jsonify({
            "total_detections": 0,
            "new_detections": 0,
            "existing_detections": 0,
            "high_confidence_wildfire": 0,
            "possible_fire": 0,
            "likely_false_positive": 0,
            "max_frp_mw": 0,
        })

    return jsonify({
        "total_detections": int(len(df)),
        "new_detections": int((df["fire_status"] == "new").sum()),
        "existing_detections": int((df["fire_status"] == "existing").sum()),
        "high_confidence_wildfire": int(
            (df["final_label"] == "high_confidence_wildfire").sum()
        ),
        "possible_fire": int(
            (df["final_label"] == "possible_fire").sum()
        ),
        "likely_false_positive": int(
            (df["final_label"] == "low_confidence_or_false_positive").sum()
        ),
        "max_frp_mw": float(df["frp"].max()) if "frp" in df.columns else 0,
    })


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    initialize_database()
    load_model()

    socketio.run(
        app,
        debug=True,
        host="0.0.0.0",
        port=5000,
        allow_unsafe_werkzeug=True,
    )