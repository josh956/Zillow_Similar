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
    df = pd.DataFrame(points_list)
    df['Date'] = pd.to_datetime(df['x'], unit='ms')
    df['Year'] = df['Date'].dt.year
    df.rename(columns={'y': 'Value'}, inplace=True)
    
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
        <div style='display:inline-block; padding:10px 14px; border-radius:8px; background:rgba(0,0,0,0.03); margin:3px; min-width:120px;'>
            <div style='font-size:13px; color:#6e6e6e;'>{label}</div>
            <div style='font-size:24px; font-weight:bold; color:{color};'>{value}</div>
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

# Check if we have comparison metrics to display
# Previously these were displayed at the top, but we're removing them to save space
# The comparison metrics will still be available in the sidebar and in property cards
# comparison_analysis available here if needed for future use

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

# Comparison address input
comp_address_input = st.sidebar.text_input("Comparison Address (optional)")

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

# Prepare cache key for current filters
current_filter_hash = get_filter_hash(st.session_state.filters)

# If cache is missing or filters changed, recalculate and cache
if (
    'avg_history_cache' not in st.session_state or
    st.session_state.get('last_filter_hash') != current_filter_hash
):
    # Get all properties with addresses (DETERMINISTIC - always the same for the filter)
    all_apts_with_address = [p for p in props if p.get("address")]
    
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
            st.image(p.get("imgSrc"), use_container_width=True)
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

                        # KPI grid
                        kpi_cols = st.columns(3)

                        # Selected metrics (blue)
                        with kpi_cols[0]:
                            render_kpi("Sel. Latest", analysis['metrics']['latest_price'], "#1E90FF")
                            render_kpi("Sel. Price Hikes", analysis['metrics']['price_hikes'], "#1E90FF")
                            render_kpi("Sel. Total Inc.", f"{analysis['metrics']['total_increase']}%", "#1E90FF")
                            render_kpi("Sel. Annual Inc.", f"{analysis['metrics']['avg_annual_increase']}%", "#1E90FF")

                        # Comparison metrics (orange) if available
                        if comp_metrics:
                            with kpi_cols[1]:
                                render_kpi("Comp Latest", comp_metrics['latest_price'], "#FF8C00")
                                render_kpi("Comp Price Hikes", comp_metrics['price_hikes'], "#FF8C00")
                                render_kpi("Comp Total Inc.", f"{comp_metrics['total_increase']}%", "#FF8C00")
                                render_kpi("Comp Annual Inc.", f"{comp_metrics['avg_annual_increase']}%", "#FF8C00")
                        else:
                            kpi_cols[1].markdown("""
                            <div style="text-align: center; padding: 10px; color: #888; background: #f5f5f5; border-radius: 5px; margin-top: 20px;">
                                <span style="font-size: 14px;">📌 Add a comparison address in the sidebar</span>
                            </div>
                            """, unsafe_allow_html=True)

                        # Average metrics (grey)
                        if avg_metrics:
                            with kpi_cols[2]:
                                render_kpi("Avg Latest", avg_metrics['latest_price'], "#666666")
                                render_kpi("Avg Price Hikes", avg_metrics['price_hikes'], "#666666")
                                render_kpi("Avg Total Inc.", f"{avg_metrics['total_increase']}%", "#666666")
                                render_kpi("Avg Annual Inc.", f"{avg_metrics['avg_annual_increase']}%", "#666666")
                        else:
                            kpi_cols[2].markdown("<br>")

                        # Build chart base
                        fig = px.line(
                            analysis['yearly_data'],
                            x='Year',
                            y='Average_Rent',
                            title='Rental Price History',
                            labels={'Average_Rent': 'Average Rent ($)', 'Year': 'Year'}
                        )
                        fig.update_traces(line=dict(color='royalblue', width=3), name='Selected Apartment')

                        # Comparison line
                        if comparison_analysis:
                            fig.add_scatter(
                                x=comparison_analysis['yearly_data']['Year'],
                                y=comparison_analysis['yearly_data']['Average_Rent'],
                                mode='lines',
                                name='Comparison',
                                line=dict(color='#FF8C00', width=3)
                            )

                        # Average line if available
                        if avg_other is not None and not avg_other.empty:
                            fig.add_scatter(
                                x=avg_other['Year'],
                                y=avg_other['Average_Rent'],
                                mode='lines',
                                name='Avg Other Apartments',
                                line=dict(color='#808080', width=3, dash='dash')
                            )

                        fig.update_layout(
                            legend=dict(title='Legend', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        fig.update_xaxes(dtick=1, tickformat='d')
                        st.plotly_chart(fig, use_container_width=False, width=350, height=220)

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

