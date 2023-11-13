import streamlit as st
import joblib
import plots as p
import plotly.graph_objects as go


st.set_page_config(page_title="Weather & Seasons Analysis", layout="wide", page_icon='🌤️')

initial_dataset = joblib.load('./joblib_files/dataset_initial.plk')

fig_width = 1000
fig_height = 600

st.markdown("<h1 style='text-align: center;'>🌤️ Weather & Seasons Analysis</h1>",unsafe_allow_html=True)
st.markdown("---")

st.pyplot(joblib.load('./joblib_files/density.plk'))

st.markdown("""##### 🌦️ How are temperature and bike rentals related?
            
* The darker regions represent higher densities of observations.
* The plot shows: as temperature increases, bike rentals also generally increase, with the highest density observed in the mid-temperature range.
* There's a significant concentration of rentals when the temperature is between approximately 15ºC and 30ºC.

""")

st.markdown("---")

st.plotly_chart(p.weather_situation_plot(initial_dataset, fig_width, fig_height))

st.markdown("""##### 🌤️☔❄️ How are bike rentals behaving in different weather conditions?

**Clear/Cloudy Weather:**
* This weather situation has the highest total number of bike rentals.
* It seems that people are more inclined to rent bikes when the weather is clear or partly cloudy.

**Mist/Cloudy Weather:**
* Bike rentals decrease compared to clear/cloudy weather but are still relatively high.
* Misty or cloudy conditions don't significantly deter people from renting bikes.

**Light Snow/Rain:**
* There is a noticeable drop in bike rentals during light snow or rain.
* Unfavorable weather conditions seem to impact bike rental numbers.

**Heavy Rain/Snow:**
* This weather situation has the lowest bike rental numbers.
* Heavy rain or snow substantially reduces the demand for bike rentals.
            
            
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