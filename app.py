import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime
import altair as alt
import os
from dotenv import load_dotenv

# ------------------------
# Load API key
# ------------------------
if "OPENAQ_API_KEY" in st.secrets:  # when running on Streamlit Cloud
    OPENAQ_API_KEY = st.secrets["OPENAQ_API_KEY"]
else:  # when running locally
    load_dotenv()
    OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

# ------------------------
# Fetch function
# ------------------------
def fetch_openaq(city, parameters, date_from, date_to):
    base_url = "https://api.openaq.org/v3/measurements"  # ✅ v3
    headers = {"X-API-Key": OPENAQ_API_KEY}  # ✅ add API key header
    params = {
        "city": city,
        "parameter": parameters,
        "date_from": date_from,
        "date_to": date_to,
        "limit": 1000,
        "page": 1,
        "offset": 0,
        "sort": "desc",
        "order_by": "datetime"
    }

    try:
        st.info(f"🔗 Fetching from OpenAQ v3: {base_url} with params {params}")
        resp = requests.get(base_url, headers=headers, params=params, timeout=20)

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

with st.sidebar:
    st.header("Controls")
    city = st.selectbox("Select city", ["Delhi", "Mumbai", "Chennai", "Bengaluru"])
    params_selected = st.multiselect("Pollutants", ["pm25", "pm10", "so2", "no2", "o3"], default=["pm25", "pm10"])
    date_from = st.date_input("From date", datetime(2025, 8, 4))
    date_to = st.date_input("To date", datetime(2025, 9, 3))
    fetch = st.button("Fetch Data")

if fetch:
    with st.spinner("Fetching air quality data..."):
        df_aq = fetch_openaq(
            city=city,
            parameters=params_selected,
            date_from=f"{date_from}T00:00:00+00:00",
            date_to=f"{date_to}T23:59:59+00:00",
        )

    if df_aq.empty:
        st.warning(f"No air quality data found for {city} between {date_from} and {date_to}. Try another range or pollutant.")
    else:
        st.success(f"✅ Fetched {len(df_aq)} records")

        # ------------------------
        # Data preview
        # ------------------------
        st.subheader("📋 Data Preview")
        st.dataframe(df_aq[["datetime", "parameter", "value", "unit", "lat", "lon"]].head(20))

        # ------------------------
        # Time-series chart
        # ------------------------
        if "datetime" in df_aq.columns:
            st.subheader("📈 Pollutant Trends Over Time")
            chart = (
                alt.Chart(df_aq)
                .mark_line(point=True)
                .encode(
                    x="datetime:T",
                    y="value:Q",
                    color="parameter:N",
                    tooltip=["datetime:T", "parameter:N", "value:Q", "unit:N"]
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No datetime field available for plotting trends.")

        # ------------------------
        # Map Visualization
        # ------------------------
        if "lat" in df_aq and "lon" in df_aq:
            st.subheader("🌍 Measurement Locations")
            m = folium.Map(location=[df_aq["lat"].mean(), df_aq["lon"].mean()], zoom_start=10)

            for _, row in df_aq.iterrows():
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=5,
                    popup=f"{row['parameter']} = {row['value']} {row['unit']}",
                    color="blue",
                    fill=True,
                    fill_opacity=0.6
                ).add_to(m)

            st_folium(m, width=700, height=500)
        else:
            st.info("No coordinates available for mapping.")
