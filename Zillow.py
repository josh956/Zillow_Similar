import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from openai import OpenAI
from datetime import datetime

# Fetch API keys securely
RAPIDAPI_KEY = os.getenv("RapidAPI") if os.getenv("RapidAPI") else st.secrets["rapidapi"]["key"]
OPENAI_API_KEY = os.getenv("General") if os.getenv("General") else st.secrets["General"]["key"]

client = OpenAI(api_key=OPENAI_API_KEY)

# Streamlit App
st.title("Apartment and House Rental Data Viewer")
st.write("Enter an address to fetch rental data, view yearly averages, and generate AI insights.")

# User input for the address
address = st.text_input("Address", value="Enter Apartment Address Here")

def fetch_rental_data(address):
    url = "https://zillow-com1.p.rapidapi.com/valueHistory/localRentalRates"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "zillow-com1.p.rapidapi.com"
    }
    querystring = {"address": address}
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            st.error("API request limit reached. Please try again later.")
        else:
            st.error(f"API request failed with status {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while making the API request: {e}")
    return None

def process_rental_data(data):
    if not data or 'chartData' not in data:
        return None
    
    points_list = [point for chart in data['chartData'] if 'points' in chart for point in chart['points']]
    df = pd.DataFrame(points_list)
    df['Date'] = pd.to_datetime(df['x'], unit='ms')
    df['Year'] = df['Date'].dt.year
    df.rename(columns={'y': 'Value'}, inplace=True)
    
    yearly_avg = df.groupby('Year')['Value'].mean().reset_index()
    yearly_avg.rename(columns={'Value': 'Average_Rent'}, inplace=True)
    yearly_avg['Percent_Change'] = yearly_avg['Average_Rent'].pct_change() * 100
    
    yearly_avg['Average_Rent'] = yearly_avg['Average_Rent'].round(0).astype(int)
    yearly_avg['Percent_Change'] = yearly_avg['Percent_Change'].round(0).fillna(0).astype(int).astype(str) + '%'
    yearly_avg['Year'] = yearly_avg['Year'].astype(str)
    
    return yearly_avg

def get_ai_insight(history_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert real estate analyst."},
                {"role": "user", "content": f"Analyze the following rental data trends and provide insights:\n\n{history_text}\n\nProvide a concise summary of the trends, including significant changes, potential reasons, and any forecasts based on past trends."}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error fetching AI insight: {e}"

if st.button("Fetch Data"):
    rental_data = fetch_rental_data(address)
    if rental_data:
        yearly_avg = process_rental_data(rental_data)
        if yearly_avg is not None:
            st.write("Yearly Average Rent and Percent Change:")
            st.dataframe(yearly_avg)
            
            fig = px.bar(yearly_avg, x='Year', y='Average_Rent', title="Yearly Average Rent",
                         labels={'Year': "Year", 'Average_Rent': "Average Rent ($)"})
            st.plotly_chart(fig)
            
            history_text = "\n".join([f"{row['Year']}: ${row['Average_Rent']} ({row['Percent_Change']})" for _, row in yearly_avg.iterrows()])
            ai_insight = get_ai_insight(history_text)
            st.subheader("AI-Generated Insights")
            st.write(ai_insight)
        else:
            st.error("No rental data found for the given address.")

# --- Footer ---
st.markdown(
    """
    <hr>
    <p style="text-align: center;">
    <b>Rent History Web App</b> &copy; 2025<br>
    Developed by <a href="https://www.linkedin.com/in/josh-poresky956/" target="_blank">Josh Poresky</a><br><br>
    </p>
    """,
    unsafe_allow_html=True
)

