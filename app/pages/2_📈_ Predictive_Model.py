import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_extras.app_logo import add_logo

st.set_page_config(page_title="Model - SafeHaven")

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

model = joblib.load('./ridge_modelfinaltest.sav','r')

st.subheader("Ridge Model", divider = "grey")

NB_file_path = './neighborhoods.txt'
year_built_file_path = './year_built.txt'

# Getting the neighborhood feature into a form that works as model input 
neighborhoods = pd.read_csv(NB_file_path, header=None)
neighborhoods = neighborhoods.rename(columns={0: 'Neighborhood'})
neighborhoods = neighborhoods.drop(columns=1)

neighborhood_select = pd.DataFrame({'Neighborhood': neighborhoods['Neighborhood']})

# Getting the year built feature into a form that works as model input 
year_built_data = pd.read_csv(year_built_file_path, header=None)
year_built_data = year_built_data.rename(columns={0: 'YearBuilt'})
year_built_data = year_built_data.drop(columns=1)

year_built_select = pd.DataFrame({'YearBuilt': year_built_data['YearBuilt']})

val1 = 0
val2 = 0
val3 = st.slider("Year the property was built:", min_value= 1700, max_value=2023, step=1)
val4 = st.slider("Sold price:", min_value = 50000, max_value= 2000000, step=10000)
val5 = st.selectbox("Neighborhood:",neighborhoods,format_func=lambda x: x.split('_')[-1].title())

chosen_year_built = val3

year_built_select['YearBuilt'] = np.where(year_built_select['YearBuilt'] == 
                                               chosen_year_built, chosen_year_built, 0)

chosen_neighborhood = val5

neighborhood_select['Neighborhood'] = np.where(neighborhood_select['Neighborhood'] == chosen_neighborhood, 1, 0)

if st.button("Predict"):
    feature_array = np.concatenate([
    np.array([[val1]]),       
    np.array([[val2]]),         
    year_built_select['YearBuilt'].values.reshape(1, -1),  # Add values from year_built_select
    np.array([[val4]]), 
    neighborhood_select['Neighborhood'].values.reshape(1, -1)  # Add values from neighborhood_select
], axis=1) 
    
    prediction = model.predict(feature_array)
    st.write("The predicted number of yearly crimes in the area of a $",int(val4),"property built in",int(val3), "in the", chosen_neighborhood.split('_')[-1].title(),"area is:", int(prediction),".")


