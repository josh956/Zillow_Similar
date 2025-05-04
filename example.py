import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from openai import OpenAI
from datetime import datetime
import time
import hashlib
import json
import random
import requests_cache
from functools import lru_cache, wraps

# New imports for Folium
import folium
from streamlit_folium import folium_static

# --- API Key Handling ---
RAPIDAPI_KEY = os.getenv("RapidAPI") if os.getenv("RapidAPI") else st.secrets["rapidapi"]["key"]
OPENAI_API_KEY = os.getenv("General") if os.getenv("General") else st.secrets["General"]["key"]

client = OpenAI(api_key=OPENAI_API_KEY)

# Install cache for all requests (cache expires after 10 minutes)
requests_cache.install_cache('zillow_cache', expire_after=600)

# Timed LRU cache decorator for function-level caching (e.g., 10 min)
def timed_lru_cache(seconds: int, maxsize: int = 128):
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = seconds
        func.expiration = time.time() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # Convert dictionary arguments to hashable tuples
            new_args = []
            for arg in args:
                if isinstance(arg, dict):
                    new_args.append(tuple(sorted(arg.items())))
                else:
                    new_args.append(arg)
            
            new_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    new_kwargs[key] = tuple(sorted(value.items()))
                else:
                    new_kwargs[key] = value
            
            if time.time() >= func.expiration:
                func.cache_clear()
                func.expiration = time.time() + func.lifetime
            return func(*tuple(new_args), **new_kwargs)
        return wrapped_func
    return wrapper_cache

# Exponential backoff for rate limits
@timed_lru_cache(600)
def fetch_with_backoff(url, headers, params, max_retries=5):
    delay = 1
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                wait_time = int(retry_after)
            else:
                wait_time = delay
                delay *= 2
            st.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            st.error(f"API request failed with status {response.status_code}: {response.text}")
            return None
    st.error("Max retries exceeded due to rate limiting.")
    return None

def fetch_rental_history(address):
    """Fetch rental history for an address with improved error handling and logging"""
    # Check if we already have this in the cache to avoid duplicate requests
    cache_key = f"rental_history_{address}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    url = "https://zillow-com1.p.rapidapi.com/valueHistory/localRentalRates"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
    }
    params = {"address": address}
    
    try:
        # Use fetch_with_backoff instead of direct request to handle rate limits
        data = fetch_with_backoff(url, headers, params)
        
        # Validate the data
        if data is None:
            return None
            
        if 'chartData' not in data:
            return None
            
        # Store in session state to avoid duplicate requests
        st.session_state[cache_key] = data
        return data
        
    except Exception as e:
        st.error(f"Error fetching rental history for {address}: {str(e)}")
        return None

def analyze_price_history(data):
    """Analyze price history data and return key metrics"""
    if not data or 'chartData' not in data:
        return None
    
    points_list = [point for chart in data['chartData'] if 'points' in chart for point in chart['points']]
    
    # Check if we have valid data points with expected structure
    if not points_list:
        return None
        
    # Validate the data structure before processing
    required_columns = ['x', 'y']
    if not all(col in points_list[0] for col in required_columns):
        return None
    
    df = pd.DataFrame(points_list)
    try:
        df['Date'] = pd.to_datetime(df['x'], unit='ms')
        df['Year'] = df['Date'].dt.year
        df.rename(columns={'y': 'Value'}, inplace=True)
    except Exception:
        # If any conversion fails, return None
        return None
    
    # Calculate yearly averages
    yearly_avg = df.groupby('Year')['Value'].mean().reset_index()
    yearly_avg.rename(columns={'Value': 'Average_Rent'}, inplace=True)
    yearly_avg['Percent_Change'] = yearly_avg['Average_Rent'].pct_change() * 100
    
    # Ensure years are sorted chronologically
    yearly_avg_sorted = yearly_avg.sort_values('Year')
    last_n_years = yearly_avg_sorted.copy()
    
    if len(last_n_years) < 1:
        return None
    
    # Calculate metrics (handle single year case)
    price_hikes = (last_n_years['Percent_Change'] > 0).sum() if len(last_n_years) > 1 else 0
    first_year_rent = last_n_years.iloc[0]['Average_Rent']
    last_year_rent = last_n_years.iloc[-1]['Average_Rent']
    total_percent_increase = ((last_year_rent - first_year_rent) / first_year_rent) * 100 if len(last_n_years) > 1 else 0
    avg_annual_increase = total_percent_increase / (len(last_n_years) - 1) if len(last_n_years) > 1 else 0
    
    # Format the data
    yearly_avg['Average_Rent'] = yearly_avg['Average_Rent'].round(0).astype(int)
    yearly_avg['Percent_Change'] = yearly_avg['Percent_Change'].round(0).fillna(0).astype(int).astype(str) + '%'
    yearly_avg['Year'] = yearly_avg['Year'].astype(str)
    
    return {
        'yearly_data': yearly_avg,
        'metrics': {
            'price_hikes': price_hikes,
            'total_increase': round(total_percent_increase, 1),
            'avg_annual_increase': round(avg_annual_increase, 1),
            'latest_price': f"${int(last_year_rent)}"
        }
    }

# Helper to compute metrics for any dataframe of yearly averages
def compute_avg_metrics(avg_df: pd.DataFrame):
    if avg_df is None or avg_df.empty:
        return None
    tmp = avg_df.copy()
    tmp['Year_int'] = tmp['Year'].astype(int)
    tmp = tmp.sort_values('Year_int')
    first = tmp.iloc[0]['Average_Rent']
    last = tmp.iloc[-1]['Average_Rent']
    hikes = (tmp['Average_Rent'].diff() > 0).sum()
    total_inc = ((last - first) / first * 100) if len(tmp) > 1 else 0
    annual_inc = total_inc / (len(tmp) - 1) if len(tmp) > 1 else 0
    return {
        'price_hikes': int(hikes),
        'total_increase': round(total_inc, 1),
        'avg_annual_increase': round(annual_inc, 1),
        'latest_price': f"${int(last)}"
    }

# KPI rendering function used throughout the app
def render_kpi(label:str, value, color:str="gray"):
    """Render a KPI card with colored text/background"""
    st.markdown(
        f"""
        <div style='display:inline-block; padding:6px 8px; border-radius:5px; background:rgba(0,0,0,0.03); margin:2px; min-width:80px;'>
            <div style='font-size:11px; color:#6e6e6e;'>{label}</div>
            <div style='font-size:16px; font-weight:bold; color:{color};'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def make_api_request(url, headers, params, max_retries=3, delay=2):
    """Make API request with retry mechanism"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    st.info(f"Rate limit reached. Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    st.error("API rate limit reached. Please try again in a few minutes.")
                    return None
            else:
                st.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            st.error(f"Error making API request: {str(e)}")
            return None
    return None

def get_filter_hash(filters):
    # Create a hash of the current filter set for cache key
    return hashlib.md5(json.dumps(filters, sort_keys=True).encode()).hexdigest()

# --- App Config ---
st.set_page_config(page_title="Zillow Property Explorer", layout="wide")
st.title("🏡 Zillow Property Explorer")

# Add a prominent banner for comparison address when none is set
if not st.session_state.get('comparison_address'):
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 10px 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #FF8C00; display: flex; align-items: center;">
        <span style="font-size: 24px; margin-right: 10px;">📊</span>
        <div>
            <span style="font-weight: bold; color: #333;">Add a comparison address in the sidebar</span><br>
            <span style="color: #666; font-size: 0.9em;">Compare prices and trends with any other property</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add a prominent comparison address input at the top of the page
top_cols = st.columns([3, 1])
with top_cols[0]:
    top_comp_address = st.text_input(
        "📍 Enter comparison address",
        value=st.session_state.get('comparison_address', ''),
        placeholder="e.g., 123 Main St, Boston, MA",
        help="Compare any property's rental history with your selected listings"
    )
    if top_comp_address:
        st.session_state['comparison_address'] = top_comp_address

with top_cols[1]:
    if st.session_state.get('comparison_address'):
        if st.button("🔄 Change Address", key="change_top"):
            st.session_state['comparison_address'] = ""
            st.rerun()

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

# Comparison address input in sidebar (synced with top input)
comp_address_input = st.sidebar.text_input(
    "Comparison Address (optional)",
    value=st.session_state.get('comparison_address', '')
)

if comp_address_input:
    st.session_state['comparison_address'] = comp_address_input

# Fetch comparison analysis
comparison_analysis = None
if 'comparison_address' in st.session_state and st.session_state['comparison_address']:
    comp_hist = fetch_rental_history(st.session_state['comparison_address'])
    comparison_analysis = analyze_price_history(comp_hist) if comp_hist else None

# Sidebar badge
if st.session_state.get('comparison_address'):
    st.sidebar.success(f"✓ Comparing to: {st.session_state['comparison_address'][:35]}…")

if comparison_analysis:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Comparison Metrics")
    if 'metrics' in comparison_analysis:
        cmp = comparison_analysis['metrics']
        render_kpi("Latest Price", cmp['latest_price'], "#FF8C00")
        render_kpi("Price Hikes", cmp['price_hikes'], "#FF8C00")
        render_kpi("Total Inc.", f"{cmp['total_increase']}%", "#FF8C00")
        render_kpi("Annual Inc.", f"{cmp['avg_annual_increase']}%", "#FF8C00")
    else:
        st.sidebar.warning("No metrics available for comparison property")

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
headers = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
}
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

results = make_api_request(url, headers, querystring)
if not results:
    st.stop()

# --- Display Results ---
props = results.get("props", [])
if not props:
    st.warning("No listings found for your criteria.")
    st.stop()

# Filter out properties with undisclosed locations or missing geographic data
props = [p for p in props if p.get("address") and 
         "undisclosed" not in p.get("address", "").lower() and
         p.get("latitude") and p.get("longitude")]

if not props:
    st.warning("Found only properties with undisclosed locations. Try different search criteria.")
    st.stop()

# Limit displayed apartments to 15
props = props[:15]

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
    zoom_start=15,
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

# Prepare cache key for current filters
current_filter_hash = get_filter_hash(st.session_state.filters)

# If cache is missing or filters changed, recalculate and cache
if (
    'avg_history_cache' not in st.session_state or
    st.session_state.get('last_filter_hash') != current_filter_hash
):
    # Get all properties with addresses (DETERMINISTIC - always the same for the filter)
    all_apts_with_address = props  # Already filtered above
    
    # Use up to 7 properties for the average calculation
    apts_for_avg = all_apts_with_address[:7]
    
    other_histories = []
    progress = st.progress(0, text="Loading average price histories...")
    
    for i, prop in enumerate(apts_for_avg):
        try:
            # Using the same approach as Zillow.py to fetch and process data
            url = "https://zillow-com1.p.rapidapi.com/valueHistory/localRentalRates"
            headers = {
                "X-RapidAPI-Key": RAPIDAPI_KEY,
                "X-RapidAPI-Host": "zillow-com1.p.rapidapi.com"
            }
            querystring = {"address": prop["address"]}
            
            # Use the fetch_with_backoff function to handle rate limits
            data = fetch_with_backoff(url, headers, querystring)
            
            if data and 'chartData' in data:
                points_list = [point for chart in data['chartData'] if 'points' in chart for point in chart['points']]
                if points_list:
                    df = pd.DataFrame(points_list)
                    df['Date'] = pd.to_datetime(df['x'], unit='ms')
                    df['Year'] = df['Date'].dt.year
                    df.rename(columns={'y': 'Value'}, inplace=True)
                    
                    yearly_avg = df.groupby('Year')['Value'].mean().reset_index()
                    yearly_avg.rename(columns={'Value': 'Average_Rent'}, inplace=True)
                    yearly_avg['Year'] = yearly_avg['Year'].astype(str)
                    
                    # Only add if we have actual data
                    if not yearly_avg.empty:
                        other_histories.append(yearly_avg)
                        st.session_state[f"apt_history_{prop['address']}"] = yearly_avg  # Cache individual histories
                else:
                    st.warning(f"No price points found for {prop['address']}")
            else:
                st.warning(f"No chart data found for {prop['address']}")
        except Exception as e:
            st.warning(f"Error fetching history for {prop['address']}: {str(e)}")
        
        progress.progress((i+1)/len(apts_for_avg), text=f"Loaded {i+1} of {len(apts_for_avg)} histories...")
    
    progress.empty()
    
    # Calculate the average across all histories - this is stored PER FILTER SET
    if other_histories:
        all_years = pd.concat(other_histories)
        avg_other = all_years.groupby('Year')['Average_Rent'].mean().reset_index()
        avg_other['Average_Rent'] = avg_other['Average_Rent'].round(0).astype(int)
        avg_other['Year'] = avg_other['Year'].astype(str)
        
        # Compute metrics across full average dataset
        avg_calc = avg_other.copy()
        avg_calc['Year_int'] = avg_calc['Year'].astype(int)
        avg_calc = avg_calc.sort_values('Year_int')

        first_year_rent = avg_calc.iloc[0]['Average_Rent']
        last_year_rent = avg_calc.iloc[-1]['Average_Rent']

        price_hikes = (avg_calc['Average_Rent'].diff() > 0).sum()
        total_percent_increase = ((last_year_rent - first_year_rent) / first_year_rent) * 100 if len(avg_calc) > 1 else 0
        avg_annual_increase = total_percent_increase / (len(avg_calc) - 1) if len(avg_calc) > 1 else 0

        avg_metrics = {
            'price_hikes': int(price_hikes),
            'total_increase': round(total_percent_increase, 1),
            'avg_annual_increase': round(avg_annual_increase, 1),
            'latest_price': f"${int(last_year_rent)}"
        }

        st.session_state['avg_metrics'] = avg_metrics
    else:
        st.warning("Could not calculate average price history. No valid data found for any properties.")
        avg_other = None
        st.session_state['avg_metrics'] = None
    
    # Store in session state - this will be the SAME for all apartments with this filter
    st.session_state['avg_history_cache'] = avg_other
    st.session_state['last_filter_hash'] = current_filter_hash
else:
    # Use the cached average (same for all apartments with this filter)
    avg_other = st.session_state['avg_history_cache']

for idx, p in enumerate(props):
    with st.container():
        cols = st.columns([1, 3])
        with cols[0]:
            # Handle missing images gracefully
            img_src = p.get("imgSrc")
            if img_src and isinstance(img_src, str) and img_src.startswith(("http://", "https://")):
                try:
                    st.image(img_src, use_container_width=True)
                except Exception:
                    st.info("Image unavailable")
            else:
                st.info("No image available")
        with cols[1]:
            st.markdown(f"### {p.get('address')}")
            
            # Ribbon if comparison enabled
            if st.session_state.get('comparison_address'):
                st.markdown(
                    f"<div style='background:#FFF4E5; color:#CC5F00; padding:4px 8px; border-radius:4px; font-size:12px; display:inline-block;'>Comparing to: {st.session_state['comparison_address'][:35]}…</div>",
                    unsafe_allow_html=True,
                )
            
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
            
            # Add the 'View on Zillow' link above the visual
            if p.get("detailUrl"):
                st.markdown(
                    f"[View on Zillow](https://www.zillow.com{p['detailUrl']})",
                    unsafe_allow_html=True
                )
            
            # Add price history analysis
            if p.get("address"):
                with st.expander("View Price History Analysis", expanded=False):
                    # Add a spinner while loading
                    with st.spinner("Loading price history..."):
                        # Fetch selected property history
                        history_data = fetch_rental_history(p["address"])
                        
                        # Debug: Show data or error
                        if history_data is None:
                            st.error(f"Could not fetch price history for {p['address']}. The API returned no data.")
                        elif 'chartData' not in history_data:
                            st.error(f"The API response for {p['address']} is missing chart data.")
                        elif not any('points' in chart for chart in history_data['chartData']):
                            st.error(f"No price points found in the chart data for {p['address']}.")
                        
                        analysis = analyze_price_history(history_data) if history_data else None
                        
                    # Show metrics and chart
                    if analysis:
                        # Gather reference metrics
                        avg_metrics = compute_avg_metrics(avg_other) if avg_other is not None else None
                        comp_metrics = comparison_analysis['metrics'] if comparison_analysis else None

                        # Create new two-column layout with KPIs on left, chart on right
                        st.markdown("<h4 style='margin-bottom: 15px;'>Rental Price History Analysis</h4>", unsafe_allow_html=True)
                        
                        # Define apartment names clearly for all displays
                        this_apt_name = p.get('address', '').split(',')[0]
                        comparison_apt_name = st.session_state.get('comparison_address', '').split(',')[0] if st.session_state.get('comparison_address') else None
                        area_name = st.session_state.filters['location']
                        
                        # Define consistent colors
                        this_color = "#1E90FF"  # Blue
                        comp_color = "#FF8C00"  # Orange
                        avg_color = "#666666"   # Gray
                        
                        analysis_cols = st.columns([2, 3])
                        
                        # Left column: KPIs with clear labels
                        with analysis_cols[0]:
                            # Get values for the metrics
                            apt_latest = analysis['metrics']['latest_price']
                            apt_hikes = analysis['metrics']['price_hikes']
                            apt_total = f"{analysis['metrics']['total_increase']}%"
                            apt_annual = f"{analysis['metrics']['avg_annual_increase']}%"
                            
                            comp_latest = comp_metrics['latest_price'] if comp_metrics else '-'
                            comp_hikes = comp_metrics['price_hikes'] if comp_metrics else '-'
                            comp_total = f"{comp_metrics['total_increase']}%" if comp_metrics else '-'
                            comp_annual = f"{comp_metrics['avg_annual_increase']}%" if comp_metrics else '-'
                            
                            avg_latest = avg_metrics['latest_price'] if avg_metrics else '-'
                            avg_hikes = avg_metrics['price_hikes'] if avg_metrics else '-'
                            avg_total = f"{avg_metrics['total_increase']}%" if avg_metrics else '-'
                            avg_annual = f"{avg_metrics['avg_annual_increase']}%" if avg_metrics else '-'
                            
                            # Apply CSS styling
                            st.markdown("""
                            <style>
                            .metric-table { width: 100%; border-collapse: separate; border-spacing: 0 8px; margin-bottom: 10px; }
                            .metric-table th { text-align: left; padding: 4px; font-size: 12px; color: #666; border-bottom: 1px solid #eee; }
                            .metric-table td { padding: 6px 4px; font-size: 14px; font-weight: bold; }
                            .left-label { padding-left: 5px; border-left: 4px solid #ccc; font-weight: normal; }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            # Build HTML table manually with f-strings
                            table_html = f"""
                            <table class="metric-table">
                            <tr>
                                <th>Metric</th>
                                <th style="color:{this_color};">{this_apt_name}</th>
                                <th style="color:{comp_color};">{comparison_apt_name if comparison_apt_name else ''}</th>
                                <th style="color:{avg_color};">Area Avg</th>
                            </tr>
                            <tr>
                                <td class="left-label" style="border-left-color:{this_color};">Latest Price</td>
                                <td style="color:{this_color};">{apt_latest}</td>
                                <td style="color:{comp_color};">{comp_latest}</td>
                                <td style="color:{avg_color};">{avg_latest}</td>
                            </tr>
                            <tr>
                                <td class="left-label" style="border-left-color:{this_color};">Price Hikes</td>
                                <td style="color:{this_color};">{apt_hikes}</td>
                                <td style="color:{comp_color};">{comp_hikes}</td>
                                <td style="color:{avg_color};">{avg_hikes}</td>
                            </tr>
                            <tr>
                                <td class="left-label" style="border-left-color:{this_color};">Total Inc.</td>
                                <td style="color:{this_color};">{apt_total}</td>
                                <td style="color:{comp_color};">{comp_total}</td>
                                <td style="color:{avg_color};">{avg_total}</td>
                            </tr>
                            <tr>
                                <td class="left-label" style="border-left-color:{this_color};">Annual Inc.</td>
                                <td style="color:{this_color};">{apt_annual}</td>
                                <td style="color:{comp_color};">{comp_annual}</td>
                                <td style="color:{avg_color};">{avg_annual}</td>
                            </tr>
                            </table>
                            """
                            st.markdown(table_html, unsafe_allow_html=True)
                        
                        if not comp_metrics:
                            st.markdown("""
                            <div style="padding: 10px; background: #f8f9fa; border-radius: 5px; margin-top: 10px; border-left: 3px solid #FF8C00;">
                                <span style="font-size: 13px;">📌 Add a comparison address at the top of the page</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Right column: Chart
                        with analysis_cols[1]:
                            # Build chart with consistent styling
                            fig = px.line(
                                analysis['yearly_data'],
                                x='Year',
                                y='Average_Rent',
                                labels={'Average_Rent': 'Average Rent ($)', 'Year': 'Year'}
                            )
                            
                            # Clear titles
                            this_apt_title = f"This property: {this_apt_name}"
                            comp_apt_title = f"Comparison: {comparison_apt_name}" if comparison_apt_name else "Comparison"
                            area_avg_title = f"Area Average ({area_name})"
                            
                            # This apartment line
                            fig.update_traces(
                                line=dict(color=this_color, width=3),
                                name=this_apt_title,
                                mode='lines+markers',
                                marker=dict(size=7, symbol='circle')
                            )

                            # Comparison line
                            if comp_metrics:
                                fig.add_scatter(
                                    x=comparison_analysis['yearly_data']['Year'],
                                    y=comparison_analysis['yearly_data']['Average_Rent'],
                                    name=comp_apt_title,
                                    line=dict(color=comp_color, width=3),
                                    mode='lines+markers',
                                    marker=dict(size=7, symbol='square')
                                )

                            # Average line if available
                            if avg_metrics:
                                fig.add_scatter(
                                    x=avg_other['Year'],
                                    y=avg_other['Average_Rent'],
                                    mode='lines+markers',
                                    name=area_avg_title,
                                    line=dict(color=avg_color, width=2, dash='dash'),
                                    marker=dict(size=6, symbol='triangle-up', color=avg_color)
                                )

                            # Better layout
                            fig.update_layout(
                                margin=dict(l=10, r=10, t=10, b=40),
                                height=280,
                                showlegend=True,
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=-0.15,
                                    xanchor='center',
                                    x=0.5,
                                    bgcolor='rgba(255,255,255,0.9)',
                                ),
                                plot_bgcolor='rgba(248,249,250,0.5)',
                                xaxis=dict(
                                    showgrid=True,
                                    gridcolor='rgba(0,0,0,0.05)',
                                ),
                                yaxis=dict(
                                    showgrid=True,
                                    gridcolor='rgba(0,0,0,0.05)',
                                    title=dict(text='Average Rent ($)', font=dict(size=12)),
                                )
                            )
                            
                            fig.update_xaxes(dtick=1, tickformat='d')
                            st.plotly_chart(fig, use_container_width=True)

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

