import streamlit as st
import pandas as pd
import requests
from datetime import date, timedelta

# ----------------------------------
# CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Smart City Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

OPENAQ_ENDPOINT = "https://api.openaq.org/v2/measurements"

# Predefined city coordinates for fallback queries
CITY_COORDS = {
    "Delhi": (28.7041, 77.1025),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
}


# ----------------------------------
# HELPERS
# ----------------------------------
@st.cache_data(show_spinner=False)
def fetch_openaq(city: str, parameters: list[str], date_from: str, date_to: str,
                 max_pages: int = 8, per_page: int = 1000) -> pd.DataFrame:
    """
    Fetch measurements from OpenAQ v2 with pagination + fallback to coordinates radius query.
    Returns dataframe or empty df. Debug info is shown in Streamlit.
    """
    if not parameters:
        return pd.DataFrame()

    rows = []
    page = 1

    # Primary: query by city name
    base_params = {
        "city": city,
        "parameter": ",".join(parameters),
        "date_from": date_from,
        "date_to": date_to,
        "limit": per_page,
        "sort": "asc",
        "order_by": "date",
    }

    st.write("OpenAQ Query (city):", base_params)

    while page <= max_pages:
        params = base_params.copy()
        params["page"] = page
        try:
            r = requests.get(OPENAQ_ENDPOINT, params=params, timeout=30)
        except Exception as e:
            st.error(f"OpenAQ request failed: {e}")
            break
        if page == 1:
            st.write("OpenAQ HTTP status:", r.status_code)

        if r.status_code != 200:
            try:
                st.write("OpenAQ response (error):", r.text[:1000])
            except Exception:
                pass
            break
        payload = r.json()
        res = payload.get("results", [])
        if not res:
            break
        rows.extend(res)
        page += 1

    # If primary city query returned nothing, try fallback with coordinates
    if not rows:
        st.info("City-level query returned no results. Trying fallback with coordinates + radius.")
        coords = CITY_COORDS.get(city)
        if coords:
            lat, lon = coords
            fallback_params = {
                "coordinates": f"{lat},{lon}",
                "radius": 50000,
                "parameter": ",".join(parameters),
                "date_from": date_from,
                "date_to": date_to,
                "limit": per_page,
                "sort": "asc",
                "order_by": "date",
            }
            st.write("OpenAQ Fallback Query (coords + radius):", fallback_params)
            page = 1
            while page <= max_pages:
                params = fallback_params.copy()
                params["page"] = page
                try:
                    r = requests.get(OPENAQ_ENDPOINT, params=params, timeout=30)
                except Exception as e:
                    st.error(f"OpenAQ fallback request failed: {e}")
                    break
                if page == 1:
                    st.write("OpenAQ fallback HTTP status:", r.status_code)
                if r.status_code != 200:
                    try:
                        st.write("OpenAQ fallback response (error):", r.text[:1000])
                    except Exception:
                        pass
                    break
                payload = r.json()
                res = payload.get("results", [])
                if not res:
                    break
                rows.extend(res)
                page += 1

    if not rows:
        st.write("OpenAQ: no rows found after both city and fallback queries.")
        return pd.DataFrame()

    # Normalize into dataframe
    df = pd.json_normalize(rows)
    if "date.utc" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date.utc"], errors="coerce")
    elif "date.local" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date.local"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    if "coordinates.latitude" in df.columns and "coordinates.longitude" in df.columns:
        df["lat"] = pd.to_numeric(df["coordinates.latitude"], errors="coerce")
        df["lon"] = pd.to_numeric(df["coordinates.longitude"], errors="coerce")

    keep = ["timestamp", "parameter", "value", "unit", "lat", "lon", "location", "country", "city"]
    df_out = df[[c for c in keep if c in df.columns]].dropna(subset=["timestamp"]).sort_values("timestamp")

    st.write("OpenAQ: total measurements fetched:", len(df_out))
    st.dataframe(df_out.head(5))  # small preview
    return df_out


# ----------------------------------
# APP LAYOUT
# ----------------------------------
st.title("🌆 Smart City Dashboard")

with st.sidebar:
    st.header("Controls")
    city = st.selectbox("Select city", list(CITY_COORDS.keys()))
    parameters = st.multiselect(
        "Pollutants", ["pm25", "pm10", "so2", "no2", "co", "o3"], default=["pm25", "pm10"]
    )
    date_to = date.today()
    date_from = date_to - timedelta(days=30)
    date_from = st.date_input("From date", date_from)
    date_to = st.date_input("To date", date_to)

    run = st.button("Fetch Data")

if run:
    with st.spinner("Fetching air quality from OpenAQ ..."):
        df_aq = fetch_openaq(city, parameters, str(date_from), str(date_to))

    # --- Debug info ---
    st.write("DEBUG: number of AQ records:", 0 if df_aq is None else len(df_aq))
    if df_aq is not None and not df_aq.empty:
        st.write("DEBUG: available parameters in returned data:", sorted(df_aq["parameter"].unique().tolist()))

    if df_aq is None or df_aq.empty:
        st.warning("No data found for this selection.")
    else:
        st.success("Data loaded!")
        st.line_chart(df_aq, x="timestamp", y="value", color="parameter")
