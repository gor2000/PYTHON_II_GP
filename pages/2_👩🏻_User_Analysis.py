import pickle

import streamlit as st
import plots as p

## Title Tab
st.set_page_config(page_title="Users Analysis", layout='wide', page_icon='👯‍♀️')

st.markdown("<h1 style='text-align: center;'>👩🏻User Analysis🧔🏼</h1>", unsafe_allow_html=True)

with open("./joblib_files/dataset_initial.plk", "rb") as file:
    initial_dataset = pickle.load(file)

fig_width = 1000  # 1344 pixels
fig_height = 600  # 576 pixels

st.markdown("---")

st.plotly_chart(p.registered_vs_casual_plotly(initial_dataset, fig_width, fig_height))

st.markdown("""##### 👩🏻 How do our Bike Rentals vary between Registered and Casual Users?

* Registered users are clearly the majority of our users, with a total of 81.2% of the rentals.
* Throughout the year, we can clearly see that summer months get the most casual users, likely do to a lot more people coming to Washington D.C. for vacations.
* We can also see with this visualization how overall demand increased during 2012 compared to 2011, for both registered and casual users.
    
""", unsafe_allow_html=True)

st.markdown("---") 

st.plotly_chart(p.hourly_distribution_plotly(initial_dataset, fig_width, fig_height))

st.markdown("""##### 🕒 How do our Bike Rentals vary throughout the day among types of Users?
            
* Interestingly, we can see that the distributions are very similar to the Working Day / Non-Working Day distributions we saw earlier, as registered useres define this patterns through their rutines.
            
**Registered Users:**
   * There are two clear peaks for registered users: one in the morning (around 7-9 am) and another in the evening (around 5-7 pm). This pattern aligns with typical commuting hours, suggesting that many registered users use bike-sharing for daily commuting.
    
**Casual Users:**
   * Casual users display a different pattern. Their usage starts increasing from the late morning and peaks in the early afternoon, then gradually decreases as the evening approaches. This indicates that casual users might be using the bikes more for leisure rather than regular commuting.

""", unsafe_allow_html=True)

st.markdown("---")

#Line Chart
st.plotly_chart(p.avg_bike_rental_hour(initial_dataset, fig_width, fig_height))
            
st.markdown("""
##### 🌞 At what time of the day do we have the most bike rentals?

* Working Days:
    * There are two peaks: one in the morning around 8 AM and another in the evening around 5-6 PM. This pattern likely corresponds to work/school commute hours.
    * The demand is relatively lower during mid-day hours.
* Non-Working Days:
    * The demand for bikes rises gradually from the morning and peaks around 1-3 PM.
    * The pattern suggests a more relaxed and dispersed use throughout the day.

""", unsafe_allow_html=True)

st.markdown("---")

#Circular Average
st.plotly_chart(p.circular_avg(initial_dataset, fig_width, fig_height))

st.markdown("""
##### 🚴🏻‍♀️ How does the whole hourly demand look?

* As we have observed before, commuting home hours between 5pm and 6pm, have the overall peak of rental bikes.

""", unsafe_allow_html=True)

st.markdown("---")

st.plotly_chart(p.plotly_daily_trends(initial_dataset, fig_width, fig_height))

st.markdown("""
##### 📈 How do daily demand trends look like?
* We observe a clear seasonal pattern, with peaks during the warmer months and troughs during colder months, consistent with our previous analyses.
* There's an upward trend from 2011 to 2012, indicating an increase in bike rentals over the years.
* There are some sporadic drops in rentals, possibly due to specific events, extreme weather conditions, or other external factors.
""", unsafe_allow_html=True)

st.markdown("---")

st.plotly_chart(p.monthly_distribution(initial_dataset, fig_width, fig_height))

st.markdown("""
##### 📆 How has the demand change between 2011 and 2012?

* The demand for bikes generally increases from January to June, peaking around the summer months (June/July).
* There's a decline in demand from July onwards, with the lowest demand during the winter months (December/January).
* We can see the surprising peak overall for both years happened in September 2012, with an average demand of 300 Bike Rentals per Hour.
* Comparing the two years, 2012 generally had a higher demand than 2011 across all months. This could be due to increased popularity of the service, marketing efforts, or other external factors.

""", unsafe_allow_html=True)

st.markdown("---")

st.plotly_chart(p.plotly_monthly_trends(initial_dataset, 1100, fig_height))

st.markdown("""
##### 🌻 In what months did we have the most rentals comparing 2011 and 2012?
* The seasonality in bike rentals is evident, with higher rentals during warmer months and lower during colder months.
* For nearly every month, 2012 shows an increase in bike rentals compared to 2011.
""", unsafe_allow_html=True)

st.markdown("---")


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 