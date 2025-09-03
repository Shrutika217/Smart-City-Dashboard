from __future__ import annotations

import io
import math
import time
import json
import joblib
import folium
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ------------------------------
# Utilities
# ------------------------------

CITY_COORDS = {
    # Add more as you like
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "London": (51.5074, -0.1278),
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Paris": (48.8566, 2.3522),
    "Singapore": (1.3521, 103.8198),
}

OPENAQ_ENDPOINT = "https://api.openaq.org/v2/measurements"

@st.cache_data(show_spinner=False)
def fetch_openaq(city: str, parameters: list[str], date_from: str, date_to: str, max_pages: int = 8, per_page: int = 1000) -> pd.DataFrame:
    """Fetch measurements from OpenAQ v2 with basic pagination.
    Returns a dataframe with columns incl. parameter, value, unit, coordinates.latitude/longitude, location, date.utc.
    """
    rows = []
    page = 1
    while page <= max_pages:
        params = {
            "city": city,
            "parameter": ",".join(parameters),
            "date_from": date_from,
            "date_to": date_to,
            "limit": per_page,
            "page": page,
            "sort": "asc",
            "order_by": "date",
        }
        r = requests.get(OPENAQ_ENDPOINT, params=params, timeout=60)
        if r.status_code != 200:
            break
        payload = r.json()
        res = payload.get("results", [])
        if not res:
            break
        rows.extend(res)
        page += 1
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    # Normalize timestamp column
    if "date.utc" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date.utc"], errors="coerce")
    elif "date.local" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date.local"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT
    # Normalize lat/lon
    if "coordinates.latitude" in df.columns and "coordinates.longitude" in df.columns:
        df["lat"] = pd.to_numeric(df["coordinates.latitude"], errors="coerce")
        df["lon"] = pd.to_numeric(df["coordinates.longitude"], errors="coerce")
    # Keep essential columns
    keep = [
        "timestamp", "parameter", "value", "unit", "lat", "lon", "location", "country", "city"
    ]
    return df[[c for c in keep if c in df.columns]].dropna(subset=["timestamp"]) \
             .sort_values("timestamp")


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df[ts_col].dt.hour
    df["dow"] = df[ts_col].dt.dayofweek
    df["month"] = df[ts_col].dt.month
    return df


def build_lagged_target(df_hourly: pd.DataFrame, target_col: str = "pm25") -> pd.DataFrame:
    """Given an hourly series with column `target_col`, create lag features and drop NaNs."""
    df = df_hourly.copy()
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    df = add_time_features(df, ts_col="timestamp")
    return df.dropna()


def train_regressor(df_feat: pd.DataFrame, target_col: str = "pm25"):
    features = [c for c in df_feat.columns if c not in ("timestamp", target_col)]
    X = df_feat[features]
    y = df_feat[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
    }
    return model, features, metrics


def recursive_forecast_next_24h(model, last_row: pd.Series, features: list[str]) -> pd.DataFrame:
    """Given a trained model and the last known feature row, perform a naive recursive 24-step forecast.
    We roll the lag features forward each step. Assumes features include lags + hour/dow/month.
    """
    preds = []
    cur = last_row.copy()
    cur_timestamp = last_row["timestamp"]
    for step in range(1, 25):
        # increment timestamp by 1 hour
        cur_timestamp = cur_timestamp + timedelta(hours=1)
        cur["timestamp"] = cur_timestamp
        cur["hour"] = cur_timestamp.hour
        cur["dow"] = cur_timestamp.dayofweek
        cur["month"] = cur_timestamp.month
        # predict
        X = cur[features].values.reshape(1, -1)
        yhat = float(model.predict(X)[0])
        preds.append({"timestamp": cur_timestamp, "pm25_pred": yhat})
        # shift lag features forward (lag_1 becomes the new prediction)
        # lags: lag_1, lag_2, lag_3, lag_6, lag_12, lag_24
        # update in descending order to avoid overwriting values too early
        for lag_name in ["lag_24", "lag_12", "lag_6", "lag_3", "lag_2", "lag_1"]:
            lag_k = int(lag_name.split("_")[-1])
            if lag_k == 1:
                cur[lag_name] = yhat
            else:
                prev = f"lag_{lag_k - 1}"
                if prev in cur:
                    cur[lag_name] = cur[prev]
    return pd.DataFrame(preds)


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Smart City Dashboard", page_icon="🚦", layout="wide")

st.title("🚦🌆 Smart City Dashboard – Traffic & Air Quality")
st.caption("End‑to‑end portfolio project: data collection → modeling → interactive maps → forecasting")

with st.sidebar:
    st.header("Settings")
    city = st.selectbox("City", options=sorted(CITY_COORDS.keys()), index=0)
    latlon = CITY_COORDS[city]

    today = datetime.now(timezone.utc).date()
    default_start = today - timedelta(days=7)
    date_from = st.date_input("Date from", value=default_start)
    date_to = st.date_input("Date to", value=today)

    st.markdown("**Parameters (Air Quality)**")
    params_selected = st.multiselect(
        "Select pollutants", options=["pm25", "pm10", "no2", "o3"], default=["pm25", "pm10"]
    )

    st.divider()
    st.subheader("Traffic data (optional)")
    traffic_file = st.file_uploader("Upload traffic CSV", type=["csv"])  # columns: timestamp, lat, lon, congestion_index

# ------------------------------
# Fetch & display air quality data
# ------------------------------

st.subheader("Air Quality – Raw Measurements")

with st.spinner("Fetching air quality from OpenAQ ..."):
    df_aq = fetch_openaq(
        city=city,
        parameters=params_selected,
        date_from=f"{date_from}T00:00:00+00:00",
        date_to=f"{date_to}T23:59:59+00:00",
    )

if df_aq.empty:
    st.warning("No air quality data returned for this city/date range/parameters. Try expanding the range or changing pollutants.")
else:
    st.write(f"Fetched **{len(df_aq):,}** measurements across **{df_aq['location'].nunique()}** stations.")
    st.dataframe(df_aq.head(20), use_container_width=True)

    # Map of latest measurements per station for a chosen parameter
    st.markdown("### Map – Latest per Station (choose parameter)")
    param_for_map = st.selectbox("Parameter to map", options=params_selected, index=0)

    # pick latest reading per station for this parameter
    latest = df_aq[df_aq["parameter"] == param_for_map].sort_values("timestamp").groupby("location").tail(1)
    latest = latest.dropna(subset=["lat", "lon"]).copy()

    m = folium.Map(location=list(latlon), zoom_start=11, control_scale=True)
    # add markers
    for _, r in latest.iterrows():
        popup = folium.Popup(
            f"<b>{r['location']}</b><br>{r['parameter'].upper()}: {r['value']} {r['unit']}<br>{r['timestamp']}",
            max_width=250
        )
        folium.CircleMarker(location=[r["lat"], r["lon"]], radius=7, fill=True, fill_opacity=0.8, popup=popup).add_to(m)

    # optional heatmap based on values
    heat_df = latest[["lat", "lon", "value"]].dropna()
    if not heat_df.empty:
        HeatMap(heat_df.values.tolist(), radius=18, blur=28, max_zoom=12).add_to(m)

    st_folium(m, width=None, height=520)

# ------------------------------
# Build hourly series & forecast PM2.5
# ------------------------------

st.subheader("PM2.5 Forecast – Next 24 Hours (Sklearn model with lag features)")

def build_hourly_pm25(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df[df["parameter"] == "pm25"].copy()
    if d.empty:
        return pd.DataFrame()
    d = d.set_index("timestamp").sort_index()
    # hourly mean across stations
    hourly = d["value"].resample("1H").mean().to_frame(name="pm25").dropna()
    hourly.reset_index(inplace=True)
    return hourly

if df_aq.empty or "pm25" not in params_selected:
    st.info("Add PM2.5 to parameters to enable forecasting.")
else:
    hourly = build_hourly_pm25(df_aq)
    if hourly.empty or len(hourly) < 48:
        st.warning("Not enough hourly PM2.5 data to train a model (need at least ~48 points). Try expanding the date range.")
    else:
        feat = build_lagged_target(hourly, target_col="pm25")
        model, features, metrics = train_regressor(feat, target_col="pm25")
        st.write("**Model Performance (holdout)**:")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{metrics['MAE']:.2f}")
        c2.metric("RMSE", f"{metrics['RMSE']:.2f}")
        c3.metric("R²", f"{metrics['R2']:.2f}")

        # Prepare last feature row for recursion
        last_row = feat.iloc[-1].copy()
        last_row["timestamp"] = pd.to_datetime(last_row["timestamp"])  # ensure Timestamp

        fcst = recursive_forecast_next_24h(model, last_row, features)
        st.write("**Next 24 hours forecast (PM2.5, µg/m³)**")
        st.dataframe(fcst, use_container_width=True, height=240)

        # Plot history + forecast
        st.markdown("**Chart – History & Forecast**")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(hourly["timestamp"], hourly["pm25"], label="History")
        plt.plot(fcst["timestamp"], fcst["pm25_pred"], label="Forecast")
        plt.xlabel("Time")
        plt.ylabel("PM2.5 (µg/m³)")
        plt.legend()
        st.pyplot(plt.gcf(), clear_figure=True)

        # Download button
        csv_bytes = fcst.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download PM2.5 Forecast CSV",
            data=csv_bytes,
            file_name=f"pm25_forecast_{city}_{datetime.utcnow().strftime('%Y%m%d%H%M')}.csv",
            mime="text/csv",
        )

# ------------------------------
# Traffic (optional): visualize + simple forecast if provided
# ------------------------------

st.subheader("Traffic – Visualization & Simple Next‑Hour Forecast (Optional)")

if traffic_file is None:
    st.info("Upload a traffic CSV with columns: timestamp, lat, lon, congestion_index to enable this section.")
else:
    try:
        df_tr = pd.read_csv(traffic_file)
    except Exception:
        traffic_file.seek(0)
        df_tr = pd.read_csv(traffic_file, encoding_errors="ignore")

    # Basic validation
    required_cols = {"timestamp", "lat", "lon", "congestion_index"}
    if not required_cols.issubset(set(df_tr.columns)):
        st.error(f"Traffic CSV must include columns: {', '.join(sorted(required_cols))}")
    else:
        df_tr["timestamp"] = pd.to_datetime(df_tr["timestamp"], errors="coerce")
        df_tr["lat"] = pd.to_numeric(df_tr["lat"], errors="coerce")
        df_tr["lon"] = pd.to_numeric(df_tr["lon"], errors="coerce")
        df_tr["congestion_index"] = pd.to_numeric(df_tr["congestion_index"], errors="coerce")
        df_tr = df_tr.dropna(subset=["timestamp", "lat", "lon", "congestion_index"]).sort_values("timestamp")

        st.write(f"Loaded **{len(df_tr):,}** traffic rows across **{df_tr[['lat','lon']].drop_duplicates().shape[0]}** sensors.")
        st.dataframe(df_tr.head(20), use_container_width=True)

        # Map latest congestion per sensor
        latest_tr = df_tr.sort_values("timestamp").groupby(["lat", "lon"]).tail(1)
        mt = folium.Map(location=list(latlon), zoom_start=11, control_scale=True)
        for _, r in latest_tr.iterrows():
            pop = folium.Popup(
                f"<b>Congestion</b>: {r['congestion_index']:.2f}<br>{r['timestamp']}",
                max_width=220,
            )
            folium.CircleMarker([r["lat"], r["lon"]], radius=7, fill=True, fill_opacity=0.8, popup=pop).add_to(mt)
        # Heatmap
        hm = latest_tr[["lat", "lon", "congestion_index"]].dropna()
        if not hm.empty:
            HeatMap(hm.values.tolist(), radius=18, blur=28, max_zoom=12).add_to(mt)
        st_folium(mt, width=None, height=520)

        # Build hourly mean congestion for simple forecast
        tr_hourly = (
            df_tr.set_index("timestamp")["congestion_index"].resample("1H").mean().to_frame("congestion")
            .dropna()
            .reset_index()
        )

        if len(tr_hourly) < 48:
            st.info("Not enough hourly traffic data to train a model (need at least ~48 points).")
        else:
            # Lag features
            tr_feat = tr_hourly.copy()
            for lag in [1, 2, 3, 6, 12, 24]:
                tr_feat[f"lag_{lag}"] = tr_feat["congestion"].shift(lag)
            tr_feat["hour"] = tr_feat["timestamp"].dt.hour
            tr_feat["dow"] = tr_feat["timestamp"].dt.dayofweek
            tr_feat["month"] = tr_feat["timestamp"].dt.month
            tr_feat = tr_feat.dropna()

            # Train/test
            X = tr_feat.drop(columns=["timestamp", "congestion"]) 
            y = tr_feat["congestion"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            m_tr = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
            m_tr.fit(X_train, y_train)
            yhat = m_tr.predict(X_test)
            mae = mean_absolute_error(y_test, yhat)
            rmse = math.sqrt(mean_squared_error(y_test, yhat))
            r2 = r2_score(y_test, yhat)
            c1, c2, c3 = st.columns(3)
            c1.metric("Traffic MAE", f"{mae:.2f}")
            c2.metric("Traffic RMSE", f"{rmse:.2f}")
            c3.metric("Traffic R²", f"{r2:.2f}")

            # Recursive next 24h for traffic
            last = tr_feat.iloc[-1].copy()
            last_ts = last.name if isinstance(last.name, pd.Timestamp) else tr_feat.iloc[-1]["timestamp"]
            if not isinstance(last_ts, pd.Timestamp):
                last_ts = pd.to_datetime(tr_feat.iloc[-1]["timestamp"])
            preds = []
            cur = last.copy()
            cur_ts = last_ts
            tr_features = list(X.columns)
            for step in range(1, 25):
                cur_ts = cur_ts + timedelta(hours=1)
                cur["hour"] = cur_ts.hour
                cur["dow"] = cur_ts.dayofweek
                cur["month"] = cur_ts.month
                Xcur = cur[tr_features].values.reshape(1, -1)
                yhat = float(m_tr.predict(Xcur)[0])
                preds.append({"timestamp": cur_ts, "congestion_pred": yhat})
                # roll lags
                for lname in ["lag_24", "lag_12", "lag_6", "lag_3", "lag_2", "lag_1"]:
                    k = int(lname.split("_")[-1])
                    if k == 1:
                        cur[lname] = yhat
                    else:
                        prev = f"lag_{k-1}"
                        if prev in cur:
                            cur[lname] = cur[prev]
            tr_fcst = pd.DataFrame(preds)

            st.write("**Traffic – Next 24 hours forecast**")
            st.dataframe(tr_fcst, use_container_width=True, height=240)

            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(tr_hourly["timestamp"], tr_hourly["congestion"], label="History")
            plt.plot(tr_fcst["timestamp"], tr_fcst["congestion_pred"], label="Forecast")
            plt.xlabel("Time")
            plt.ylabel("Congestion Index")
            plt.legend()
            st.pyplot(plt.gcf(), clear_figure=True)

            st.download_button(
                "Download Traffic Forecast CSV",
                tr_fcst.to_csv(index=False).encode("utf-8"),
                file_name=f"traffic_forecast_{city}_{datetime.utcnow().strftime('%Y%m%d%H%M')}.csv",
                mime="text/csv",
            )

# ------------------------------
# Footer / Tips
# ------------------------------

st.divider()
st.markdown(
    """
**Tips to extend this project**  
• Join **weather** features (humidity, wind, rain) from Open‑Meteo or OpenWeather; add to lag feature set  
• Swap RandomForest with **XGBoost/LightGBM** or **Prophet** for stronger temporal patterns  
• Add **model explainability** (SHAP) to show which lags/hours drive predictions  
• Deploy to **Streamlit Community Cloud** or **Render/AWS/GCP** and link it on your CV  
• Add city selector on map + sensor filtering by parameter & threshold  
    """
)
