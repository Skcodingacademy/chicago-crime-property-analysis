import streamlit as st
import pandas as pd
from streamlit_extras.app_logo import add_logo

st.set_page_config(page_title="Home - SafeHaven", layout = "wide", page_icon = "🏡", initial_sidebar_state="expanded")

developers_markdown = """
Seth Kulow ([GitHub](https://github.com/Skcodingacademy))\n
Israel Dellinger ([GitHub](https://github.com/idellinger))\n 
Rashid Baset ([GitHub](https://github.com/rbaset5))\n
"""

def logo():
    add_logo("./resources/images/logo.png", height=250)
logo()

st.sidebar.title("Developers")
st.sidebar.info(developers_markdown)
st.image("./resources/images/logo_notext.png")
st.title("SafeHaven Analytics")
st.subheader("A predictive model for housing analysis", divider="grey")

# Home page #
@st.cache_data # Cached the DataFrame so it only has to load once
def load_data_head(filepath):
    df = pd.read_csv(filepath)
    df = df.head()
    return df
df_house = load_data_head("./resources/data/HomeHarvest_Chicago.csv")
df_crime = load_data_head("./resources/data/CrimeData_Chicago.csv")

intro_markdown = """
Using publicly accessible data from the City of Chicago and Realtor.com (via HomeHarvest), we developed a machine learning model to demonstrate the relationships between neighborhoods, housing, and crime.
"""
st.write(intro_markdown)


with st.expander("HomeHarvest Database Example"):
    st.dataframe(df_house)

with st.expander("City of Chicago Crime Database Example"):
    st.dataframe(df_crime)

body_markdown = """
Our project aims to forecast the number of crimes in Chicago neighborhoods by considering different factors related to housing.\n
These include where the homes are located, how much they were last sold for, and the year in which were built.\n
Unlike typical models that predict housing prices, our focus is on exploring how information about homes might provide insights into the number of crimes in the city of Chicago.
"""
st.write(body_markdown)
# Images on the Home page #
tab1, tab2, tab3, tab4 = st.tabs(["Violent Crime","Property Crime","Days on Market","Crime Rate by Year"]) # Image tabs

with tab1:
    st.subheader("Violent crime by neighborhood")
    st.image("./resources/images/violent_cloro.png")
    
with tab2:
    st.subheader("Property crime by neighborhood")
    st.image("./resources/images/property_cloro.png")

with tab3:
    st.subheader("Average days on market by neighborhood")
    st.image("./resources/images/daysonmarket.png")

with tab4:
    st.subheader("Change in crime rate by year (%) ")
    st.image("./resources/images/crime_rate_year.png")
