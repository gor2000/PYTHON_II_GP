import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd


def dist_total(df, bins):
    # Create a histogram using Plotly
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df['cnt'],
            nbinsx=bins,
            marker_color='skyblue',
            opacity=0.7,
        )
    )

    # Update layout
    fig.update_layout(
        title='Distribution of Total Bike Rentals',
        xaxis_title='Number of Bike Rentals',
        yaxis_title='Frequency',
        showlegend=False,
        bargap=0.1  # Adjust for bin separation
    )

    return fig



# fig_width = 14 * 96  # 1344 pixels
# fig_height = 6 * 96  # 576 pixels
# fig = plots.avg_bike_rental_hour(df, fig_width, fig_height)



def avg_bike_rental_hour(df, fig_width, fig_height):
    # Group by hour and workingday, calculate the mean for cnt
    grouped_data = df.groupby(['hr', 'workingday']).cnt.mean().reset_index()

    # Create the figure
    fig = go.Figure()

    # Line plot for workingday=0 (non-working day)
    fig.add_trace(go.Scatter(x=grouped_data[grouped_data['workingday'] == 0]['hr'],
                             y=grouped_data[grouped_data['workingday'] == 0]['cnt'],
                             mode='lines', name='Non-Working Day', line=dict(color='dodgerblue', backoff=0.7)))

    # Line plot for workingday=1 (working day)
    fig.add_trace(go.Scatter(x=grouped_data[grouped_data['workingday'] == 1]['hr'],
                             y=grouped_data[grouped_data['workingday'] == 1]['cnt'],
                             mode='lines', name='Working Day', line=dict(color='MediumSlateBlue', backoff=0.7)))

    # Update layout
    fig.update_layout(
        title='Average Bike Rentals per Hour by Working Day',
        xaxis=dict(title='Hour of the Day'),
        yaxis=dict(title='Average Number of Bike Rentals'),
        legend_title='Working Day',
        width=fig_width, height=fig_height
    )

    return fig



# fig_width = 14 * 96  # 1344 pixels
# fig_height = 6 * 96
# fig = plots.circular_avg(df, fig_width, fig_height)
def circular_avg(df, fig_width, fig_height):
    hourly_counts = df.groupby("hr")["cnt"].mean().reset_index()
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    # Create a polar bar plot

    fig = go.Figure(go.Barpolar(
        r=hourly_counts["cnt"],
        theta=np.degrees(theta),  # Convert radians to degrees for Plotly

        marker_color=hourly_counts["cnt"],  # Color based on count value
        marker_colorscale='Jet',  # Color scale similar to Matplotlib's 'jet'
        opacity=0.6
    ))

    fig.update_layout(
        title='Circular Area Plot for Hourly Distribution of Bike Rentals',
        polar=dict(
            radialaxis=dict(showticklabels=False, ticks=''),
            angularaxis=dict(tickvals=np.degrees(theta), ticktext=np.arange(24))
        ),
        showlegend=False,
        width=fig_width, height=fig_height
    )
    return fig



def monthly_distribution(df, fig_width, fig_height):
    # Group by month and year, then calculate the mean
    monthly_avg = df.groupby(['mnth', 'yr'])['cnt'].mean().reset_index()

    # Create a Plotly figure
    fig = go.Figure()

    # Line plot for 2011
    fig.add_trace(go.Scatter(x=monthly_avg[monthly_avg['yr'] == 0]['mnth'],
                             y=monthly_avg[monthly_avg['yr'] == 0]['cnt'],
                             mode='lines', name='2011',
                             line=dict(color='darkslateblue', backoff=0.7)))

    # Line plot for 2012
    fig.add_trace(go.Scatter(x=monthly_avg[monthly_avg['yr'] == 1]['mnth'],
                             y=monthly_avg[monthly_avg['yr'] == 1]['cnt'],
                             mode='lines', name='2012',
                             line=dict(color='darkturquoise', backoff=0.7)))

    # Update layout for a better appearance
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ),
        title='Monthly Distribution of Bike Rentals for 2011 and 2012',
        xaxis_title='Month',
        yaxis_title='Average Number of Hourly Bike Rentals',
        legend_title='Year',
        width=fig_width, height=fig_height  # Set figure size
    )

    return fig



# fig = plots.density_plot(df, (14, 6))
# plt.show()
def density_plot(df, figsize):
    # De-normalize the temperatures
    df['actual_temp'] = df['temp'] * 41

    # Create a figure and axes object
    fig, ax = plt.subplots(figsize=figsize)

    # Use Seaborn's function with the axes object
    sns.kdeplot(data=df, x='actual_temp', y='cnt', cmap='viridis', fill=True, ax=ax)

    # Set the title and labels
    ax.set_title('Density Plot: Relationship between Actual Temperature (Celsius) and Bike Rentals')
    ax.set_xlabel('Actual Temperature (Celsius)')
    ax.set_ylabel('Number of Bike Rentals')

    # Remove the temporary column
    del df['actual_temp']

    # Return the figure object
    return fig


def plotly_daily_trends(df, fig_width, fig_height):
    # Create a copy of the DataFrame and add the calculated columns

    # Group by date and sum the bike rentals
    daily_trends = df.groupby('dteday').sum()['cnt']

    # Create a line plot with Plotly
    fig = px.line(daily_trends, x=daily_trends.index, y=daily_trends,
                  title='Daily Trends of Bike Rentals',
                  labels={'x': 'Date', 'y': 'Total Bike Rentals'})

    # Update layout for a better appearance
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Total Bike Rentals',
        width=fig_width, height=fig_height  # Set figure size
    )

    return fig


# fig = plots.plotly_monthly_trends(aux_data, 1600, 600)  # 16x6 inches at 100 PPI
def plotly_monthly_trends(df, fig_width, fig_height):
    # Create a copy of the DataFrame and perform the necessary grouping
    monthly_trends = df.groupby([df['dteday'].dt.year.rename('year'),
                                 df['dteday'].dt.month.rename('month')])['cnt'].sum().reset_index()

    # Format the 'year, month' for the x-axis
    monthly_trends['year_month'] = monthly_trends['month'].astype(str) + '/' + monthly_trends['year'].astype(str)

    # Create a bar plot with Plotly
    fig = px.bar(monthly_trends, x='year_month', y='cnt',
                 title='Monthly Trends of Bike Rentals for 2011 and 2012',
                 labels={'year_month': 'Year, Month', 'cnt': 'Total Bike Rentals'},
                 color='cnt',
                 color_continuous_scale='Blues')

    # Update layout for a better appearance
    fig.update_layout(
        xaxis_title='Month / Year',
        yaxis_title='Total Bike Rentals',
        width=fig_width, height=fig_height,  # Set figure size
        xaxis={'tickangle': 45}  # Rotate x-axis labels
    )

    return fig


# fig = plots.plotly_seasonal_trends(aux_data, 1000, 600)  # 10x6 inches at 100 PPI
def plotly_seasonal_trends(aux_data, fig_width, fig_height):
    seasonal_trends = aux_data.groupby('season')['cnt'].mean().reset_index()

    # Define season labels
    seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    seasonal_trends['season'] = seasonal_trends['season'].map(seasons)

    # Create a bar plot with Plotly
    fig = px.bar(seasonal_trends, x='season', y='cnt',
                 title='Average Bike Rentals across Seasons',
                 labels={'season': 'Season', 'cnt': 'Average Bike Rentals'},
                 color='cnt',
                 color_continuous_scale='blues'
                 #text='cnt'
                )
                     
    # Update layout for a better appearance
    fig.update_layout(
        xaxis_title='Season',
        yaxis_title='Average Bike Rentals',
        width=fig_width, height=fig_height  # Set figure size
    )

    return fig


# fig = plots.bike_rental_distribution(aux_data, 1600, 1200)  # Adjust width and height as needed
def bike_rental_distribution(aux_data, fig_width, fig_height):
    # Create a copy of the DataFrame for the necessary calculations

    # Create a subplot layout with 2 rows and 2 columns
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Holidays vs. Non-Holidays', 'Across Weekdays'))

    # Plot 1: Holidays vs. Non-Holidays
    holiday_trends = aux_data.groupby('holiday')['cnt'].mean()
    fig.add_trace(go.Bar(x=['Non-Holiday', 'Holiday'], y=holiday_trends, marker_color=['lightseagreen', 'skyblue']), row=1, col=1)

    # Plot 2: Across Weekdays
    weekday_trends = aux_data.groupby('weekday')['cnt'].mean()
    days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    fig.add_trace(go.Bar(x=days, y=weekday_trends, marker_color='steelblue'), row=2, col=1)

    # Update layout
    fig.update_layout(title_text='Distribution of Bike Rentals across Different Categorical Features', height=fig_height, width=fig_width)
    fig.update_layout(showlegend=False)

    return fig


# fig = plots.temp_vs_bike_rentals_plotly(aux_data, 1500, 600)  # Adjust width and height as needed
def temp_vs_bike_rentals_plotly(df, fig_width, fig_height):
    # Define season labels and colors
    season_details = {
        1: {'label': 'Winter', 'color': 'steelblue'},
        2: {'label': 'Spring', 'color': 'mediumseagreen'},
        3: {'label': 'Summer', 'color': 'gold'},
        4: {'label': 'Fall', 'color': 'tomato'}
    }

    # Create a subplot layout with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Actual Temperature vs. Bike Rentals', 'Feeling Temperature vs. Bike Rentals'))

    # Add scatter plots for each season
    for season, details in season_details.items():
        season_df = df[df['season'] == season]
        fig.add_trace(
            go.Scatter(x=season_df['aux_actual_temp'], y=season_df['cnt'],
                       mode='markers', marker=dict(size=5, color=details['color'], opacity=0.5),
                       name=details['label'], showlegend=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=season_df['aux_actual_atemp'], y=season_df['cnt'],
                       mode='markers', marker=dict(size=5, color=details['color'], opacity=0.5),
                       name=details['label'], showlegend=False),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        title_text='Temperature vs. Bike Rentals',
        height=fig_height, width=fig_width
    )
    fig.update_xaxes(title_text='Actual Temperature (Celsius)', row=1, col=1)
    fig.update_yaxes(title_text='Bike Rentals', row=1, col=1)
    fig.update_xaxes(title_text='Feeling Temperature (Celsius)', row=1, col=2)
    fig.update_yaxes(title_text='Bike Rentals', row=1, col=2)

    return fig


def humidity_wind_speed_vs_bike_rentals_plotly(df, fig_width, fig_height):
    # Define season labels and colors
    season_details = {
        1: {'label': 'Winter', 'color': 'steelblue'},
        2: {'label': 'Spring', 'color': 'mediumseagreen'},
        3: {'label': 'Summer', 'color': 'gold'},
        4: {'label': 'Fall', 'color': 'tomato'}
    }

    # Create a subplot layout with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Humidity vs. Bike Rentals', 'Wind Speed vs. Bike Rentals'))

    for season, details in season_details.items():
        season_df = df[df['season'] == season]
        fig.add_trace(
            go.Scatter(x=(season_df['hum']*100), y=season_df['cnt'],
                       mode='markers', marker=dict(size=5, color=details['color'], opacity=0.5),
                       name=details['label'], showlegend=(season == 0)),  # Show legend only for the first season
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=(season_df['windspeed']*67), y=season_df['cnt'],
                       mode='markers', marker=dict(size=5, color=details['color'], opacity=0.5),
                       name=details['label'], showlegend=True),
            row=1, col=2
        )

    fig.update_layout(
        title_text='Humidity & Wind Speed vs. Bike Rentals',
        height=fig_height, width=fig_width
    )
    fig.update_xaxes(title_text='Humidity', row=1, col=1)
    fig.update_yaxes(title_text='Bike Rentals', row=1, col=1)
    fig.update_xaxes(title_text='Wind Speed', row=1, col=2)
    fig.update_yaxes(title_text='Bike Rentals', row=1, col=2)

    return fig


def casual_vs_registered_histogram_plotly(df, fig_width, fig_height):
    # Create the figure
    fig = go.Figure()

    # Histogram for casual users
    fig.add_trace(go.Histogram(x=df['casual'], name='Casual Users',
                               marker_color='orange', opacity=0.7,
                               xbins=dict(size=df['casual'].max()/50),  # Bins size based on the range of data
                               histnorm=''))

    # Histogram for registered users
    fig.add_trace(go.Histogram(x=df['registered'], name='Registered Users',
                               marker_color='lightblue', opacity=0.5,
                               xbins=dict(size=df['registered'].max()/50),  # Bins size based on the range of data
                               histnorm=''))

    # Update layout
    fig.update_layout(
        title_text='Distribution of Casual vs. Registered Users',
        xaxis_title='Number of Users',
        yaxis_title='Frequency',
        barmode='overlay',  # Overlay the histograms
        width=fig_width, height=fig_height
    )

    return fig


def hourly_distribution_plotly(df, fig_width, fig_height):
    # Group by hour and calculate the mean for casual and registered users
    hourly_data = df.groupby('hr')[['casual', 'registered']].mean().reset_index()

    # Create the figure
    fig = go.Figure()

    # Line plot for casual users
    fig.add_trace(go.Scatter(x=hourly_data['hr'], y=hourly_data['casual'],
                             mode='lines', name='Casual Users',
                             line=dict(color='DodgerBlue')))

    # Line plot for registered users
    fig.add_trace(go.Scatter(x=hourly_data['hr'], y=hourly_data['registered'],
                             mode='lines', name='Registered Users',
                             line=dict(color='MediumSlateBlue')))

    # Update layout
    fig.update_layout(
        title='Hourly Distribution of Casual vs. Registered Users',
        xaxis=dict(title='Hour of the Day'),
        yaxis=dict(title='Average Number of Bike Rentals'),
        width=fig_width, height=fig_height
    )

    return fig


def correlation_matrix_plot(df, fig_width, fig_height):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Exclude specified columns in the copy
    df_copy = df_copy.drop(columns=['aux_actual_temp','aux_actual_atemp'], errors='ignore')

    corr_matrix = df_copy.corr()

    # Extract column and index names
    labels = corr_matrix.index
    values = corr_matrix.values

    # Create a heatmap trace
    heatmap_trace = go.Heatmap(
        z=values,
        x=labels,
        y=labels,
        colorscale='Viridis',
        colorbar=dict(title="Correlation Matrix"),
    )

    # Create the layout
    layout = go.Layout(
        title='Correlation Matrix',
        xaxis=dict(tickangle=-45),
        yaxis=dict(tickangle=45),
        width=fig_width,
        height=fig_height
    )

    # Create the figure
    fig = go.Figure(data=[heatmap_trace], layout=layout)

    return fig


def boxplot_outliers(df, fig_width, fig_height):
    # Create a boxplot trace
    boxplot_trace = px.box(
        df,
        y=['temp', 'hum', 'windspeed'],
        title="Boxplot with Outliers for Weather Variables",
        width=fig_width,
        height=fig_height,
        points="outliers", 
        color_discrete_sequence=px.colors.sequential.Viridis
    )

    return boxplot_trace

def boxplot_cnt_outliers(df, fig_width, fig_height):
    # Create a boxplot trace
    boxplot_trace = px.box(
        df,
        y='cnt',
        title="Boxplot with Count Outliers",
        width=fig_width,
        height=fig_height,
        points="outliers", 
        color_discrete_sequence=['mediumseagreen']
    )

    return boxplot_trace



def dist_total_var(df, col, xaxis, yaxis, title, bins):
    # Create a histogram using Plotly
    fig = go.Figure()

    # Add histogram trace
    fig.add_trace(
        go.Histogram(
            x=df[col],
            nbinsx=bins,
            marker_color='skyblue',
            opacity=0.7,
            name='Temperature Distribution'
        )
    )

    # Calculate average 'cnt' for each temperature bin
    avg_cnt_by_temp = df.groupby(pd.cut(df[col], bins=bins))['cnt'].mean()

    # Add a line for average 'cnt' in each bin
    #fig.add_trace(go.Scatter(
        #x=avg_cnt_by_temp.index.categories.mid,
        #y=avg_cnt_by_temp.values,
        #mode='lines',
        #line=dict(color='red', width=2),
        #name='Average Bike Count'))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        showlegend=False,
        bargap=0.1  # Adjust for bin separation
    )

    return fig


def temperature_seasons(df, temperature_col='aux_actual_temp', season_col='season'):
    # Define season labels
    seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}

    # Map season labels to the DataFrame
    df['season_label'] = df[season_col].map(seasons)

    # Create a box plot using Plotly Express
    fig = px.box(df, x='season_label', y=temperature_col, color='season_label',
                 title='Temperature Distribution Among Seasons',
                 labels={'season_label': 'Season', temperature_col: 'Temperature (Celsius)'},
                 width=1100, height=600)
    
    fig.update_layout(showlegend=False)

    return fig


def registered_vs_casual_plotly(df, fig_width, fig_height):
    # Create a copy of the dataframe to avoid modifying the original dataset
    df_copy = df.copy()

    # Replace 0 and 1 with 2011 and 2012 in the 'yr' column only in the copied dataframe
    df_copy['yr'] = df_copy['yr'].replace({0: 2011, 1: 2012})

    # Combine 'yr' and 'mnth' into a new column 'year_month'
    df_copy['year_month'] = df_copy['mnth'].astype(str) + '/' + df_copy['yr'].astype(str)

    # Use Plotly Express to create an interactive bar chart
    fig = px.bar(df_copy, x='year_month', y=['casual', 'registered'],
                 color_continuous_scale='viridis',
                 labels={'value': 'Number of Bike Rentals', 'variable': 'User Type'},
                 title='Registered vs Casual Users Bike Rentals by Month/Year')

    # Set figure size
    fig.update_layout(width=fig_width, height=fig_height)

    # Adjust opacity
    fig.update_traces(opacity=0.7)

    # Change x-axis label
    fig.update_layout(xaxis_title='Month / Year')

    return fig


def weather_situation_plot(df, fig_width, fig_height):

    weather_trends = df.groupby('weathersit')['cnt'].mean().reset_index()
    weather_labels = ['Clear/Cloudy', 'Mist/Cloudy', 'Light Snow/Rain', 'Heavy Rain/Snow']

    print(weather_trends)

    # Define colors for each weather situation
    colors = ['rgb(253, 231, 37)', 'rgb(144, 195, 32)', 'rgb(32, 144, 141)', 'rgb(70, 51, 126)']


    # Create a figure
    fig = go.Figure()

    # Add bar trace with different colors for each category
    fig.add_trace(go.Bar(x=weather_labels, y=weather_trends['cnt'], marker_color=colors, showlegend=False))

    # Update layout
    fig.update_layout(
        title='Bike Rentals Across Different Weather Conditions',
        xaxis_title='Weather Conditions',
        yaxis_title='Average Hourly Bike Rentals',
        width=fig_width,
        height=fig_height
    )

    return fig
    


import plotly.graph_objects as go

def plot_predictions(elements, y_test, title, pred):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(elements)), y=y_test.values[:elements],
                             mode='lines+markers', name='Actual', line=dict(color='skyblue'),
                             marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=list(range(elements)), y=pred[:elements],
                             mode='lines+markers', name='Predicted',
                             line=dict(color='orangered', dash='dashdot', width=1.5),
                             marker=dict(size=3)))

    fig.update_layout(title=f'Actual vs Predicted Bike Rentals ({title})',
                      xaxis_title='Sample',
                      yaxis_title='Number of Bike Rentals',
                      legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
                      showlegend=True,
                      height=700, width=1000)

    return fig


















