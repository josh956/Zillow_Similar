import os
import streamlit as st
import requests
import pandas as pd

# New imports for Folium
import folium
from streamlit_folium import folium_static

# --- API Key Handling ---
RAPIDAPI_KEY = os.getenv("RapidAPI") if os.getenv("RapidAPI") else st.secrets["rapidapi"]["key"]

# --- App Config ---
st.set_page_config(page_title="Zillow Property Explorer", layout="wide")
st.title("🏡 Zillow Property Explorer")

# --- Sidebar Filters ---
st.sidebar.header("Filter Listings")

# Store filter values in session state
if 'filters' not in st.session_state:
    st.session_state.filters = {
        'location': "02113",
        'status_type': "ForRent",
        'home_type': [],
        'sort': "Payment_Low_High",
        'rent_min': 0,
        'rent_max': None,
        'beds_min': 0,
        'beds_max': None,
        'baths_min': 0,
        'baths_max': None
    }

# Create filter inputs
location = st.sidebar.text_input("Location (Zip or City)", st.session_state.filters['location'])
status_type = st.sidebar.selectbox("Listing Type", ["ForRent", "ForSale", "RecentlySold"], 
                                 index=["ForRent", "ForSale", "RecentlySold"].index(st.session_state.filters['status_type']))
home_type = st.sidebar.multiselect("Home Type", [
    "Houses", "Apartments", "Townhomes", "Condos", "Multi-family", "Manufactured", "LotsLand"
], default=st.session_state.filters['home_type'])
sort = st.sidebar.selectbox("Sort By", [
    "Payment_Low_High", "Verified_Source", "Payment_High_Low", "Newest",
    "Bedrooms", "Bathrooms", "Square_Feet", "Lot_Size"
], index=["Payment_Low_High", "Verified_Source", "Payment_High_Low", "Newest",
          "Bedrooms", "Bathrooms", "Square_Feet", "Lot_Size"].index(st.session_state.filters['sort']))
rent_min = st.sidebar.number_input("Rent Min Price", 0, 20000, st.session_state.filters['rent_min'], 100)
rent_max = st.sidebar.number_input("Rent Max Price", 0, 50000, st.session_state.filters['rent_max'] or 0, 100)
beds_min = st.sidebar.number_input("Min Beds", 0, 10, st.session_state.filters['beds_min'], 1)
beds_max = st.sidebar.number_input("Max Beds", 0, 10, st.session_state.filters['beds_max'] or 0, 1)
baths_min = st.sidebar.number_input("Min Baths", 0, 10, st.session_state.filters['baths_min'], 1)
baths_max = st.sidebar.number_input("Max Baths", 0, 10, st.session_state.filters['baths_max'] or 0, 1)

# Add Apply and Clear buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Apply Filters"):
        st.session_state.filters = {
            'location': location,
            'status_type': status_type,
            'home_type': home_type,
            'sort': sort,
            'rent_min': rent_min,
            'rent_max': rent_max if rent_max > 0 else None,
            'beds_min': beds_min,
            'beds_max': beds_max if beds_max > 0 else None,
            'baths_min': baths_min,
            'baths_max': baths_max if baths_max > 0 else None
        }
        st.rerun()
with col2:
    if st.button("Clear Filters"):
        st.session_state.filters = {
            'location': "02113",
            'status_type': "ForRent",
            'home_type': [],
            'sort': "Payment_Low_High",
            'rent_min': 0,
            'rent_max': None,
            'beds_min': 0,
            'beds_max': None,
            'baths_min': 0,
            'baths_max': None
        }
        st.rerun()

# --- API Call ---
url = "https://zillow-com1.p.rapidapi.com/propertyExtendedSearch"
querystring = {
    "location": st.session_state.filters['location'],
    "status_type": st.session_state.filters['status_type'],
    "home_type": ",".join(st.session_state.filters['home_type']),
    "sort": st.session_state.filters['sort'],
    "rentMinPrice": st.session_state.filters['rent_min'],
    "rentMaxPrice": st.session_state.filters['rent_max'],
    "bedsMin": st.session_state.filters['beds_min'],
    "bedsMax": st.session_state.filters['beds_max'],
    "bathsMin": st.session_state.filters['baths_min'],
    "bathsMax": st.session_state.filters['baths_max'],
    "page": 1
}
headers = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
}

try:
    response = requests.get(url, headers=headers, params=querystring)
    response.raise_for_status()
    results = response.json()
except requests.exceptions.RequestException:
    st.error("Something went wrong while fetching data. Please try again later.")
    st.stop()
except ValueError:
    st.error("Invalid response format. Please contact support.")
    st.stop()

# --- Display Results ---
props = results.get("props", [])
if not props:
    st.warning("No listings found for your criteria.")
    st.stop()

# --- Map View with Folium ---
st.subheader("🗺️ Property Map")
# Build a DataFrame for convenience
map_data = pd.DataFrame([
    {
        "lat": float(p.get("latitude", 0)),
        "lon": float(p.get("longitude", 0)),
        "address": p.get("address"),
        "price": p.get("price") or "$0",
        "detailUrl": p.get("detailUrl", "")
    }
    for p in props
    if p.get("latitude") and p.get("longitude")
])

# Create Folium map centered on the mean location
m = folium.Map(
    location=[map_data["lat"].mean(), map_data["lon"].mean()],
    zoom_start=16,
    control_scale=True
)

# Add a marker for each property
for _, row in map_data.iterrows():
    popup_html = f"""
      <b>{row['address']}</b><br>
      Price: {row['price']}<br>
      <a href="https://www.zillow.com{row['detailUrl']}" target="_blank">
        View on Zillow
      </a>
    """
    folium.Marker(
        [row["lat"], row["lon"]],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=row["address"]
    ).add_to(m)

# Render the map in Streamlit
folium_static(m, width=1000, height=600)

# --- Card View ---
st.subheader("🏘️ Listings")
for p in props:
    with st.container():
        cols = st.columns([1, 3])
        with cols[0]:
            st.image(p.get("imgSrc"), use_container_width=True)
        with cols[1]:
            st.markdown(f"### {p.get('address')}")
            st.markdown(f"**Price:** {p.get('price', 'N/A')}")
            
            # Handle bed/bath display for multi-unit properties
            if p.get("isMultiUnit", False):
                st.markdown("**Property Type:** Multi-Unit Building")
                st.markdown("ℹ️ This property contains multiple units. Click 'View on Zillow' to see all available units and their configurations.")
            else:
                beds = p.get('beds', 'N/A')
                baths = p.get('baths', 'N/A')
                if beds == 'N/A' and baths == 'N/A':
                    st.markdown("ℹ️ Multiple units—see Zillow for details")
                else:
                    st.markdown(f"**Beds:** {beds} | **Baths:** {baths}")
            
            if p.get("detailUrl"):
                st.markdown(
                    f"[View on Zillow](https://www.zillow.com{p['detailUrl']})",
                    unsafe_allow_html=True
                )

# --- Footer ---
st.markdown(
    """
    <hr>
    <p style="text-align: center;">
    <b>Zillow Property Explorer</b> &copy; 2025<br>
    Developed by <a href="https://www.linkedin.com/in/josh-poresky956/" target="_blank">
    Josh Poresky</a><br><br>
    </p>
    """,
    unsafe_allow_html=True
)

