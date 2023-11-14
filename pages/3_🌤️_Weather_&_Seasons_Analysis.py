import pickle

import streamlit as st
import plots as p
import plotly.graph_objects as go


st.set_page_config(page_title="Weather & Seasons Analysis", layout="wide", page_icon='🌤️')

with open("./joblib_files/dataset_initial.plk", "rb") as file:
    initial_dataset = pickle.load(file)

fig_width = 1000
fig_height = 600

st.markdown("<h1 style='text-align: center;'>🌤️ Weather & Seasons Analysis</h1>",unsafe_allow_html=True)
st.markdown("---")

with open("./joblib_files/density.plk", "rb") as file:
    fig_name = pickle.load(file)

st.pyplot(fig_name)
# header_image = "Images/density.png"
# st.image(header_image, width=fig_width)

st.markdown("""##### 🌦️ How are temperature and bike rentals related?
            
* The darker regions represent higher densities of observations.
* The plot shows: as temperature increases, bike rentals also generally increase, with the highest density observed in the mid-temperature range.
* There's a significant concentration of rentals when the temperature is between approximately 15ºC and 30ºC.

""")

st.markdown("---")

st.plotly_chart(p.weather_situation_plot(initial_dataset, fig_width, fig_height))

st.markdown("""##### 🌤️☔❄️ How are bike rentals behaving in different weather conditions?

**Clear/Cloudy Weather:**
* Days in Washington D.C. are in their majority within these weather conditions, with **65.67%** of total analyzed days. 
* They too constitute the largest share of bike rentals, with **71.01%** of total rentals.

**Mist/Cloudy Weather:**
* This less pleasent weather conditions represent **26.15%** of total observed days.
* Bike rentals during these conditions drop slightly, and percentage of total rentals are very consistent with their sare of days with **24.27%**.

**Light Snow/Rain:**
* These weather conditions are present for **8.17%** of total analyzed days and constitute just **4.81%** of total rentals.
* Avg Hourly Bike Rentals are **45.56%** lower than the average for Clear/Cloudy days, showing the strong impact of weather conditions in our bike rentals.

**Heavy Rain/Snow:**
* This weather situation has only been seen **3** days in the **2** years of our data.
* Heavy rain or snow days only total **233** bike rentals during the analyzed period.
            
            
**Overall, the graph suggests a correlation between weather conditions and bike rental trends, with better weather leading to increased rentals. This information can be valuable for bike rental businesses to anticipate demand based on weather forecasts.**
            
""", unsafe_allow_html=True)


st.markdown("---")

st.plotly_chart(p.temp_vs_bike_rentals_plotly(initial_dataset, fig_width, fig_height))

st.plotly_chart(p.humidity_wind_speed_vs_bike_rentals_plotly(initial_dataset, fig_width, fig_height))

st.markdown("""##### 💧 How are Temperature, Humidity and Windspeed related to our bike rentals?

**Temperature vs. Bike Rentals:**
   * There is a clear positive correlation between temperature and bike rentals, with higher temperatures associated with more bike rentals.
            
**Humidity vs. Bike Rentals:**
   * Relationship with humidity doesn't seem strongly linear, but there is a slight tendency for rentals to decrease as humidity is either too low or too high.
   * The influence of different seasons remains visible, with higher humidity often associated with summer days.
    
**Wind Speed vs. Bike Rentals:**
   * There isn't a very strong trend visible between wind speed and bike rentals. However, extremely high wind speeds have fewer bike rentals, which is understandable as high winds can make biking challenging and less safe.

""", unsafe_allow_html=True) 

st.markdown("---")


st.plotly_chart(p.plotly_seasonal_trends(initial_dataset, 1100, fig_height))

st.markdown("""
##### ⛄ How does demand behave in different seasons?
* **Winter:**  For sure, this season sees the lowest average bike rentals, this is most likely due to the cold weather conditions in Washington D.C.
* **Spring:**  Demand begins to increase relative to winter months, but still lower than the other seasons, as the weather begins to warm up.
* **Summer:**  Has the highest average bike rentals, which must align with better weather conditions and probably also attributed to more people traveling or on vacation during these months.
* **Fall:**    Demand begins to decline, but not dramatically, as the weather gets colder. Temperatures for this season are likely still pleasant enough for bike rides.

**So lets see how the weather behaves during these seasons in our next visualization.**
            
""", unsafe_allow_html=True)

st.markdown("---")

st.plotly_chart(p.temperature_seasons(initial_dataset,temperature_col='aux_actual_temp', season_col='season'))

st.markdown("""
##### 🍁 How are the temperatures during the different seasons ?
* **Winter:**  Temperatures are the lowest during this season, ranging in between 9ºC and 15ºC mostly, with just a few days over 25ºC.
* **Spring:**  Temperatures begin to rise, ranging in between 18ºC and 26ºC during most of the season.
* **Summer:**  Temperatures are the highest during this season, which shows a strong direct relation between the temperatures and the highest demand of the year.
* **Fall:**    Temperatures begin to decline, but are still quite pleasent, explaining the minor decrease in demand during this season. 
     
""", unsafe_allow_html=True)


st.markdown("---")
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
