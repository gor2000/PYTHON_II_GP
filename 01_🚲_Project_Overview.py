import pickle

import streamlit as st
import pandas as pd
import plots as p
import plotly as py
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Project Overview", layout='centered', page_icon='🚲')

st.markdown("<h1 style='text-align: center;'>🚲 Bicycle Sharing</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Group Project Python II</h4>", unsafe_allow_html=True)

st.markdown("---")

header_image = "Images/monuments-bike-tour.jpg"
st.image(header_image, width=700)  # use_container_width=True makes the image fit the width of the container

with open("./joblib_files/dataset_initial.plk", "rb") as file:
    initial_dataset = pickle.load(file)

# Data Description
st.subheader(" 🏙 Project Description 🚲")
st.write("""
    The administration of Washington D.C wants to make a deeper analysis of the usage of the bike-sharing 
    service present in the city in order to build a predictor model that helps the public transport department 
    anticipate better the provisioning of bikes in the city. For these purposes, some data is available for the years 2011 and 2012.
         """)
st.markdown("---")

st.subheader(" 👀Raw Data Overview")

st.dataframe(initial_dataset.head(10))

st.markdown("---")
st.markdown("""### 📊 Summary Statistics""", unsafe_allow_html=True)
stats = initial_dataset.drop(['dteday', 'instant'], axis=1)
st.write(stats.describe())

st.markdown("""##### 📋 Variable Observations""", unsafe_allow_html=True)
st.markdown("""

**Overall we can see we have no missing values in the dataset.**

* **season:**       The data spans across all four seasons.
* **yr:**           The dataset covers two years: 2011 and 2012.
* **mnth:**         The data spans all 12 months.
* **hr:**           The data covers all 24 hours of the day.
* **holiday:**      Most of the records are for non-holidays.
* **weekday:**      The data covers all days of the week.
* **workingday:**   About 68% of the records are from working days.
* **weathersit:**   Most days have clear or partly cloudy weather.
* **temp & atemp:** Temperatures range from 0.02 to 1 (normalized), with an average around 0.5.
* **hum:**          The average humidity is around 63%.
* **windspeed:**    Wind speeds vary, with an average around 19% (normalized).
* **casual:**       The number of casual users is significantly less than registered users on average.
* **registered:**   The number of registered users represents around 82% of total bike rentals.
* **cnt:**          The total count of bike users ranges from 1 to 977, with a median of 142.
""", unsafe_allow_html=True)

st.markdown("---")

# Data Visualization
st.header("Data Distribution")

col_name = 'aux_actual_temp'
x_axis_label = 'Temperature ºC'
y_axis_label = 'Frequency'
plot_title = 'Distribution Plot of Temperatures'
num_bins = 50

st.plotly_chart(p.dist_total_var(initial_dataset, col_name, x_axis_label, y_axis_label, plot_title, num_bins))

st.markdown("""
##### 🌡 What are the temperatures we deal with in Washington D.C?  

* We can see here that temperatures in this city can go from 0ºC to 40ºC.         
* The mean temperature throught the year is 20ºC. 
* The most common temperature in the city is around 25ºC, which is a pleasant temperature for a bike ride.
""")

st.markdown("---")

# Histogram
st.plotly_chart(p.dist_total(initial_dataset, bins=50))
fig_width = 800  # 1344 pixels
fig_height = 600  # 576 pixels

st.markdown("""
##### ⏰How many bike rentals do we have per hour?

* During most hours, the number of bike rentals per hour is less than 20.          
* The distribution is right-skewed, meaning there are many hours with a relatively low number of rentals, but there are also a few peak hours with a high number of rentals
* There are some hours with rentals exceeding 800, which might indicate peak demand times.
""")

st.markdown("---")

st.plotly_chart(p.bike_rental_distribution(initial_dataset, 700, 1000))

st.markdown("""##### 🧮 How does our data look among categories?

* In our first chart we can see that average hourly rentals are higher during Non-Holiday days, which is expected.
* In our second chart we can see that average hourly rentals across all weekdays are pretty stable with a very slight increase on Thursdays and Fridays.
* Interestingly Saturdays and Sundays have slightly lower average hourly rentals, which might indicate that people use the bikes more for commuting than for leisure. 

""", unsafe_allow_html=True)

import base64


def get_table_download_link(df):
    # Function to generate download link for CSV
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
    href = f'data:file/csv;base64,{b64}'

    # Create a button with the download link
    download_button = f'<a href="{href}" download="raw_data.csv"><button style="cursor:pointer;background-color:#4CAF50;color:white;padding:8px 12px;border:none;border-radius:4px;">Download CSV</button></a>'

    return download_button


# Download Raw Data Section
st.header("Download Raw Data")
st.write("Click the button below to download the raw data:")
st.markdown(get_table_download_link(initial_dataset), unsafe_allow_html=True)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
