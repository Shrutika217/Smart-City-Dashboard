import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime
import altair as alt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ------------------------
# Fetch available locations
# ------------------------
def fetch_locations(country="IN"):
    """Fetch available OpenAQ locations for a country (default India)."""
    url = "https://api.openaq.org/v3/locations"
    params = {"country": country, "limit": 100}
    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            st.error(f"❌ Failed to fetch locations: {resp.status_code}")
            return []
        data = resp.json()
        locations = [loc["name"] for loc in data.get("results", [])]
        return sorted(list(set(locations)))
    except Exception as e:
        st.error(f"⚠️ Error fetching locations: {e}")
        return []

# ------------------------
# Fetch measurements
# ------------------------
def fetch_openaq(location, parameters, date_from, date_to):
    base_url = "https://api.openaq.org/v3/measurements"
    params = {
        "location": location,   # ✅ v3 requires "location" not "city"
        "parameter": parameters,
        "date_from": date_from,
        "date_to": date_to,
        "limit": 1000,
        "page": 1,
        "sort": "desc",
        "order_by": "datetime"
    }

    try:
        st.info(f"🔗 Fetching from OpenAQ v3: {base_url} with params {params}")
        resp = requests.get(base_url, params=params, timeout=20)

        if resp.status_code != 200:
            st.error(f"❌ API request failed: {resp.status_code} - {resp.text}")
            return pd.DataFrame()

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Extract coordinates
        if "coordinates" in df.columns:
            df["lat"] = df["coordinates"].apply(lambda x: x.get("latitude") if isinstance(x, dict) else None)
            df["lon"] = df["coordinates"].apply(lambda x: x.get("longitude") if isinstance(x, dict) else None)

        # Parse datetime
        if "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"].apply(lambda x: x.get("utc")))

        return df

    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
        return pd.DataFrame()

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Smart City Dashboard", layout="wide")

st.title("🏙️ Smart City Dashboard")

# Sidebar controls
st.sidebar.header("Controls")

# Fetch available locations dynamically
locations = fetch_locations("IN")
location = st.sidebar.selectbox("Select monitoring station", locations if locations else ["Mumbai"])

parameters = st.sidebar.multiselect(
    "Pollutants",
    ["pm25", "pm10", "no2", "so2", "co", "o3"],
    default=["pm25"]
)

date_from = st.sidebar.text_input("From date", "2023-01-01")
date_to = st.sidebar.text_input("To date", datetime.today().strftime("%Y-%m-%d"))

if st.sidebar.button("Fetch Data"):
    df = fetch_openaq(location, parameters, date_from, date_to)

    if df.empty:
        st.warning(f"No air quality data found for {location} between {date_from} and {date_to}.")
    else:
        st.success(f"✅ Retrieved {len(df)} records for {location}")

        # Show dataframe
        st.dataframe(df.head())

        # Plot time series with Altair
        if "datetime" in df.columns and "value" in df.columns:
            chart = alt.Chart(df).mark_line().encode(
                x="datetime:T",
                y="value:Q",
                color="parameter:N"
            ).properties(width=800, height=400, title=f"Air Quality Trends - {location}")
            st.altair_chart(chart, use_container_width=True)

        # Map view
        if "lat" in df.columns and "lon" in df.columns:
            st.subheader("🌍 Monitoring Locations Map")
            m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=6)
            for _, row in df.iterrows():
                if pd.notnull(row["lat"]) and pd.notnull(row["lon"]):
                    folium.Marker(
                        location=[row["lat"], row["lon"]],
                        popup=f"{row['parameter']}: {row['value']} {row['unit']}"
                    ).add_to(m)
            st_folium(m, width=800, height=500)
