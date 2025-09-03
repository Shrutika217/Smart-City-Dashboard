import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime
import altair as alt

# ------------------------
# Fetch cities
# ------------------------
def fetch_cities(country="IN"):
    url = "https://api.openaq.org/v3/cities"
    params = {"country": country, "limit": 100}
    headers = {"X-API-Key": st.secrets["OPENAQ_API_KEY"]}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        if resp.status_code != 200:
            st.error(f"❌ Failed to fetch cities: {resp.status_code} - {resp.text}")
            return []
        data = resp.json()
        return sorted([city["name"] for city in data.get("results", []) if "name" in city])
    except Exception as e:
        st.error(f"⚠️ Error fetching cities: {e}")
        return []

# ------------------------
# Fetch locations for a city
# ------------------------
def fetch_locations(city, country="IN"):
    url = "https://api.openaq.org/v3/locations"
    params = {"country": country, "city": city, "limit": 100}
    headers = {"X-API-Key": st.secrets["OPENAQ_API_KEY"]}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        if resp.status_code != 200:
            st.error(f"❌ Failed to fetch locations: {resp.status_code} - {resp.text}")
            return []
        data = resp.json()
        return sorted([loc["name"] for loc in data.get("results", []) if "name" in loc])
    except Exception as e:
        st.error(f"⚠️ Error fetching locations: {e}")
        return []

# ------------------------
# Fetch measurements
# ------------------------
def fetch_measurements(location, parameters, date_from, date_to):
    url = "https://api.openaq.org/v3/measurements"
    params = {
        "location": location,
        "parameter": parameters,
        "date_from": date_from,
        "date_to": date_to,
        "limit": 1000,
        "page": 1,
        "offset": 0,
        "sort": "desc",
        "order_by": "datetime"
    }
    headers = {"X-API-Key": st.secrets["OPENAQ_API_KEY"]}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        if resp.status_code != 200:
            st.error(f"❌ API request failed: {resp.status_code} - {resp.text}")
            return pd.DataFrame()

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Coordinates
        if "coordinates" in df.columns:
            df["lat"] = df["coordinates"].apply(lambda x: x.get("latitude") if isinstance(x, dict) else None)
            df["lon"] = df["coordinates"].apply(lambda x: x.get("longitude") if isinstance(x, dict) else None)

        # Datetime
        if "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"].apply(lambda x: x.get("utc")))

        return df

    except Exception as e:
        st.error(f"⚠️ Error fetching measurements: {e}")
        return pd.DataFrame()

# ------------------------
# Streamlit UI
# ------------------------
st.title("🏙️ Smart City Air Quality Dashboard")

st.sidebar.header("Controls")

# Step 1: Select city
cities = fetch_cities()
selected_city = st.sidebar.selectbox("Select City", options=cities)

# Step 2: Select station in city
locations = fetch_locations(selected_city) if selected_city else []
selected_location = st.sidebar.selectbox("Select Monitoring Station", options=locations)

# Step 3: Pollutants
parameters = ["pm25", "pm10", "so2", "no2", "co", "o3", "bc"]
selected_parameters = st.sidebar.multiselect("Pollutants", options=parameters, default=["pm25", "pm10"])

# Step 4: Date range
date_from = st.sidebar.date_input("From date", datetime(2023, 1, 1))
date_to = st.sidebar.date_input("To date", datetime.today())

# Fetch button
if st.sidebar.button("Fetch Data"):
    if selected_location and selected_parameters:
        df = fetch_measurements(
            location=selected_location,
            parameters=selected_parameters,
            date_from=date_from.strftime("%Y-%m-%d"),
            date_to=date_to.strftime("%Y-%m-%d")
        )

        if df.empty:
            st.warning(f"No air quality data found for {selected_location} between {date_from} and {date_to}.")
        else:
            st.success(f"✅ Retrieved {len(df)} records")
            st.dataframe(df.head())

            # Line chart
            if "value" in df.columns:
                chart = alt.Chart(df).mark_line().encode(
                    x="datetime:T",
                    y="value:Q",
                    color="parameter:N"
                ).properties(width=700, height=400, title="Pollutant Levels Over Time")
                st.altair_chart(chart)

            # Map
            if "lat" in df.columns and df["lat"].notnull().any():
                m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=10)
                for _, row in df.iterrows():
                    if pd.notnull(row["lat"]) and pd.notnull(row["lon"]):
                        folium.CircleMarker(
                            location=[row["lat"], row["lon"]],
                            radius=4,
                            popup=f"{row['parameter']}: {row['value']} {row['unit']}",
                            color="blue",
                            fill=True,
                            fill_color="blue"
                        ).add_to(m)
                st_folium(m, width=700, height=500)
    else:
        st.warning("⚠️ Please select a city, a monitoring station, and at least one pollutant.")
