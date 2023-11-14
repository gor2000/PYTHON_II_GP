import pickle

import streamlit as st
import joblib
import plots as p
import plotly.graph_objects as go


st.set_page_config(page_title="Data Preprocessing", layout="wide", page_icon='⚙️')

with open("./joblib_files/dataset_initial.plk", "rb") as file:
    initial_dataset = pickle.load(file)

fig_width = 1000
fig_height = 600

st.markdown("<h1 style='text-align: center;'>⚙️ Data Preprocessing </h1>",unsafe_allow_html=True)
st.markdown("---")

st.plotly_chart(p.correlation_matrix_plot(initial_dataset, fig_width, fig_height))

st.markdown("""##### 📊 Correlation Matrix Observations
* Registered and casual have the highest positive correlations with total bike rentals, as expected.
* Temperature variables (temp, atemp, and their actual values) also have strong positive correlations.
* Humidity (hum) has the most significant negative correlation, indicating that as humidity increases, bike rentals tend to decrease.

""", unsafe_allow_html=True)

st.markdown("---")

st.plotly_chart(p.boxplot_outliers(initial_dataset, fig_width, fig_height))

st.markdown("""##### ⛈️ Boxplot Weather Variables Observations
            
We do not have potential outliers in *temp*. However, we have **22** potential outliers in *hum*, and **342** in *windspeed*.
            
**Temperature (temp):**
   * Both actual and feeling temperatures don't seem to have any visible outliers.
     
**Humidity (hum):**
   * There are some points below the lower whisker, indicating potential outliers with very low humidity values.
     
**Wind Speed (windspeed):**
   * There are some points above the upper whisker, indicating potential outliers with high wind speeds.
            
""", unsafe_allow_html=True)

st.markdown("---")

st.plotly_chart(p.boxplot_cnt_outliers(initial_dataset, fig_width, fig_height))

st.markdown("""##### 🚴🏻 Outliers Total Bike Rentals Observations
            
**Total Bike Rentals (cnt):**
* There are **505** observable outliers above the top whiskers, suggesting days with extremely high bike rentals.
            
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""### ⛓️ Handling Outliers

* **Humidity (hum):**  Outliers have been replaced with the median value.
* **WindSpeed (windspeed):** We created a threshold, and we applied cap and floor.
* **Total Bike Rentals (cnt):** We did the same, cap and floor creating a threshold.      
            
""",unsafe_allow_html=True)

st.code("""
        
data_outliers_handled = aux_data.copy()

median_hum = data_outliers_handled['hum'].median()
lower_bound_hum = outliers_dict['hum']['hum'].min()
upper_bound_hum = outliers_dict['hum']['hum'].max()
data_outliers_handled['hum'] = data_outliers_handled['hum'].apply(lambda x: median_hum if x < lower_bound_hum or x > upper_bound_hum else x)

lower_bound_ws = outliers_dict['windspeed']['windspeed'].min()
upper_bound_ws = outliers_dict['windspeed']['windspeed'].max()
data_outliers_handled['windspeed'] = data_outliers_handled['windspeed'].apply(lambda x: lower_bound_ws if x < lower_bound_ws else (upper_bound_ws if x > upper_bound_ws else x))

lower_bound_cnt = outliers_dict['cnt']['cnt'].min()
upper_bound_cnt = outliers_dict['cnt']['cnt'].max()
data_outliers_handled['cnt'] = data_outliers_handled['cnt'].apply(lambda x: lower_bound_cnt if x < lower_bound_cnt else (upper_bound_cnt if x > upper_bound_cnt else x))

data_outliers_handled[['hum', 'windspeed', 'cnt']].describe()
            
    """, language="python")


st.markdown("---")

st.markdown("""### 🛠️ Extra Features Generation""", unsafe_allow_html= True)

st.code("""

data_outliers_handled['dteday'] = pd.to_datetime(data_outliers_handled['dteday'])

# Extracting from date the week of the year and the quarter
data_outliers_handled['week_of_year'] = data_outliers_handled['dteday'].dt.isocalendar().week
data_outliers_handled['quarter'] = data_outliers_handled['dteday'].dt.quarter

# Creating a dictionary with the approximate duration of daylight for each month.
daylight_duration = {
    1: 10,  # January
    2: 11,  # February
    3: 12,  # March
    4: 13,  # April
    5: 14,  # May
    6: 15,  # June
    7: 15,  # July
    8: 14,  # August
    9: 13,  # September
    10: 12, # October
    11: 11, # November
    12: 10  # December
}

# Creating a new column with the approximate duration of daylight for each month.
data_outliers_handled['approx_daylight_duration'] = data_outliers_handled['mnth'].map(daylight_duration)
        
# Rush hour between 7-9 am and 5-7 pm
data_outliers_handled['rush_hour'] = data_outliers_handled['hr'].
        apply(lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0)

# Temperature difference between actual and feeling temperature.
data_outliers_handled['temp_difference'] = 
        data_outliers_handled['aux_actual_temp'] - data_outliers_handled['aux_actual_atemp']
        
    """, language="python")

with open("./joblib_files/data_outliers.plk", "rb") as file:
    data_outliers_handled = pickle.load(file)

st.dataframe(data_outliers_handled[['dteday', 'week_of_year', 'quarter', 'approx_daylight_duration','hr', 'holiday', 'rush_hour', 'temp', 'atemp', 'temp_difference']].head())

st.markdown("---")

st.markdown("""### 🧬 Advanced Features Generation""", unsafe_allow_html= True)

st.code("""

# 1. Polynomial Features
data_outliers_handled['temp_squared'] = data_outliers_handled['temp'] ** 2
data_outliers_handled['atemp_squared'] = data_outliers_handled['atemp'] ** 2
data_outliers_handled['hum_squared'] = data_outliers_handled['hum'] ** 2
data_outliers_handled['windspeed_squared'] = data_outliers_handled['windspeed'] ** 2

# 2. Quantization
# Using KBinsDiscretizer to quantize 'temp' and 'hum' into 4 bins
k_bins = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
data_outliers_handled[['temp_quantized', 'hum_quantized']] = k_bins.fit_transform(data_outliers_handled[['temp', 'hum']])

# 3. Binarization
# Binarizing 'temp' and 'hum' based on their median values
binarizer_temp = Binarizer(threshold=data_outliers_handled['temp'].median())
binarizer_hum = Binarizer(threshold=data_outliers_handled['hum'].median())

data_outliers_handled['temp_binarized'] = binarizer_temp.transform(data_outliers_handled[['temp']])
data_outliers_handled['hum_binarized'] = binarizer_hum.transform(data_outliers_handled[['hum']])
            
        """, language="python")

st.dataframe(data_outliers_handled[['temp', 'temp_squared', 'temp_quantized', 'temp_binarized', 
      'hum', 'hum_squared', 'hum_quantized', 'hum_binarized']].head())

st.markdown("---")

st.markdown("""### 🔢 Encoding""", unsafe_allow_html= True)

st.code("""

# Defining categorical and non-categorical variables
cat_vars = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
non_categorical_vars = list(set(data_outliers_handled.columns) - set(cat_vars))

# Define One Hot Encoding "model"
ohe = OneHotEncoder(sparse_output = False, drop = 'first')

# Train "model"
encoded_features = ohe.fit_transform(data_outliers_handled[cat_vars])

# "Predict"
dat_ohe = pd.DataFrame(ohe.transform(data_outliers_handled[cat_vars]))

# Get feature names for New Columns
dat_ohe.columns = ohe.get_feature_names_out()

# Combine numerical and categorical
dat = pd.concat((data_outliers_handled[non_categorical_vars], dat_ohe), axis=1)
        
        """, language="python")

st.markdown("---")

st.markdown("""### 💾 Optimizing Data Types""", unsafe_allow_html= True)   

st.code("""
def optimize_dtypes(df):
    for col in df.columns:
        col_dtype = df[col].dtype
        if col_dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_dtype == 'int64' or col_dtype == 'int32':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df
optimize_dtypes(dat)
        """, language="python")

st.markdown("---")

st.markdown("""### ✅ Final Processed Data Set""", unsafe_allow_html= True)   

data_processed = joblib.load('./joblib_files/data_processed.plk')

st.dataframe(data_processed.head())

st.markdown("---")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 