import streamlit as st
import folium
import streamlit_folium
import branca
import geopandas as gpd
from streamlit_extras.app_logo import add_logo


st.set_page_config(page_title="Map - SafeHaven", layout = "wide")

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


st.subheader("Neighborhood Crime Map (2023 Data)", divider = "grey")

crime_neighborhood_merge_path = ("./resources/data/crime_neighborhood_merge.geojson")
crime_neighborhood_merge = gpd.read_file(crime_neighborhood_merge_path)

crime_columns = ['Crimes Involving Children', 'Deceptive Practices and Fraud', 'Drug-Related Crimes',
                 'Property Crimes', 'Public Order Crimes', 'Violent Crimes']
crime_neighborhood_2023 = crime_neighborhood_merge[crime_neighborhood_merge['year'] == 2023]

# Calculate the total crimes for the year 2023
crime_neighborhood_2023['total_crime_2023'] = crime_neighborhood_2023.groupby('community')[crime_columns].transform('sum').sum(axis=1)
crime_neighborhood_2023_display = crime_neighborhood_2023.drop(columns='geometry').copy()

# Calculate quantiles for the filtered data
vmin = crime_neighborhood_2023['total_crime_2023'].quantile(0.0)
vmax = crime_neighborhood_2023['total_crime_2023'].quantile(1)

# Create the colormap
colormap = branca.colormap.LinearColormap(vmin=vmin, vmax=vmax, colors=["green", "darkgreen", "orange", "red"], caption="Total Crimes (2023)")

m = folium.Map(location=[41.84469250265398, -87.60914087617894], zoom_start=10)

popup = folium.GeoJsonPopup(
    fields=['community', 'Crimes Involving Children','Deceptive Practices and Fraud','Drug-Related Crimes','Property Crimes','Public Order Crimes','Violent Crimes'],
    aliases=['Neighborhood','Crimes Involving Children','Deceptive Practices and Fraud','Drug-Related Crimes','Property Crimes','Public Order Crimes','Violent Crimes'],
    localize=True,
    labels=True,
    style="background-color: yellow;",
)

tooltip = folium.GeoJsonTooltip(
    fields=['community', 'Crimes Involving Children','Deceptive Practices and Fraud','Drug-Related Crimes','Property Crimes','Public Order Crimes','Violent Crimes'],
    aliases=['Neighborhood','Crimes Involving Children','Deceptive Practices and Fraud','Drug-Related Crimes','Property Crimes','Public Order Crimes','Violent Crimes'],
    localize=True,
    sticky=False,
    labels=True,
    style="""
        background-color: #F0EFEF;
        border: 1px solid black;
        border-radius: 2px;
        box-shadow: 2px;
    """,
    max_width=800,
)

g = folium.GeoJson(crime_neighborhood_2023,
                   style_function=lambda x: {"fillColor": colormap(x["properties"]["total_crime_2023"])
                                             if x["properties"]["total_crime_2023"] is not None
                                             else "transparent",
                                             "color": "black",
                                             "fillOpacity": 0.4},
                   tooltip=tooltip,
                   popup=popup).add_to(m)

colormap.add_to(m)

col1, col2 = st.columns([3, 1])

with col2:
    st.image("./resources/images/logo.png")
    st.write("The map shows crime data from the year 2023 broken down by neighborhood and crime category (Violent, Property, Drug-Related, etc.)")
    with st.expander("2023 Crime Data"):
        st.dataframe(crime_neighborhood_2023_display)
with col1:
    streamlit_folium.folium_static(m, width= 1000, height=800)


