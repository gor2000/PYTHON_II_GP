import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


# p.dist_total((12, 6), initial_dataset, 50)
def dist_total(figsize, df, bins):
    sns.set_style("whitegrid")

    # Create a figure and axes object
    fig, ax = plt.subplots(figsize=figsize)

    # Use Seaborn's function with the axes object
    sns.histplot(df['cnt'], bins=bins, kde=True, color='skyblue', ax=ax)

    # Set the title and labels
    ax.set_title('Distribution of Total Bike Rentals (cnt)')
    ax.set_xlabel('Number of Bike Rentals')
    ax.set_ylabel('Frequency')

    # Return the figure object
    return fig


# fig_width = 14 * 96  # 1344 pixels
# fig_height = 6 * 96  # 576 pixels
# fig = plots.avg_bike_rental_hour(df, fig_width, fig_height)

def avg_bike_rental_hour(df, fig_width, fig_height):
    grouped_data = df.groupby(['hr', 'workingday']).cnt.mean().reset_index()

    fig = px.line(grouped_data, x='hr', y='cnt', color='workingday',
                  labels={'cnt': 'Average Number of Bike Rentals', 'hr': 'Hour of the Day',
                          'workingday': 'Working Day'},
                  title='Average Bike Rentals per Hour by Working Day')

    fig.update_layout(width=fig_width, height=fig_height,
                      xaxis_title='Hour of the Day',
                      yaxis_title='Average Number of Bike Rentals',
                      legend_title='Working Day')
    return fig


# fig_width = 14 * 96  # 1344 pixels
# fig_height = 6 * 96
# fig = plots.circular_avg(df, fig_width, fig_height)
def circular_avg(df, fig_width, fig_height):
    hourly_counts = df.groupby("hr")["cnt"].mean().reset_index()
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    fig_width = 14 * 96  # 1344 pixels
    fig_height = 6 * 96

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


# fig = plots.monthly_distribution(df, 1344, 576)
# fig.show()
def monthly_distribution(df, fig_width, fig_height):
    # Group by month and year, then calculate the mean
    monthly_avg = df.groupby(['mnth', 'yr'])['cnt'].mean().reset_index()

    # Create a line plot using Plotly
    fig = px.line(monthly_avg, x='mnth', y='cnt', color='yr',
                  labels={'mnth': 'Month', 'cnt': 'Average Number of Bike Rentals', 'yr': 'Year'},
                  title='Monthly Distribution of Bike Rentals for 2011 and 2012')

    # Update layout for a better appearance
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ),
        xaxis_title='Month',
        yaxis_title='Average Number of Bike Rentals',
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
def plotly_monthly_trends(aux_data, fig_width, fig_height):
    # Create a copy of the DataFrame and perform the necessary grouping
    monthly_trends = aux_data.groupby([aux_data['dteday'].dt.year.rename('year'),
                                       aux_data['dteday'].dt.month.rename('month')])['cnt'].sum().reset_index()

    # Format the 'year, month' for the x-axis
    monthly_trends['year_month'] = monthly_trends['year'].astype(str) + ', ' + monthly_trends['month'].astype(str)

    # Create a bar plot with Plotly
    fig = px.bar(monthly_trends, x='year_month', y='cnt',
                 title='Monthly Trends of Bike Rentals for 2011 and 2012',
                 labels={'year_month': 'Year, Month', 'cnt': 'Total Bike Rentals'},
                 color='cnt',
                 color_continuous_scale='Blues')

    # Update layout for a better appearance
    fig.update_layout(
        xaxis_title='Year, Month',
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
                 color_continuous_scale='blues',
                 text='cnt')

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
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Holidays vs. Non-Holidays', 'Across Weekdays', 'Across Weather Situations'))

    # Plot 1: Holidays vs. Non-Holidays
    holiday_trends = aux_data.groupby('holiday')['cnt'].mean()
    fig.add_trace(go.Bar(x=['Non-Holiday', 'Holiday'], y=holiday_trends, marker_color=['salmon', 'lightseagreen']), row=1, col=1)

    # Plot 2: Across Weekdays
    weekday_trends = aux_data.groupby('weekday')['cnt'].mean()
    days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    fig.add_trace(go.Bar(x=days, y=weekday_trends, marker_color='lightblue'), row=1, col=2)

    # Plot 3: Across Weather Situations
    weather_trends = aux_data.groupby('weathersit')['cnt'].mean()
    weather_labels = ['Clear/Cloudy', 'Mist/Cloudy', 'Light Snow/Rain', 'Heavy Rain/Snow']
    fig.add_trace(go.Bar(x=weather_labels, y=weather_trends, marker_color='lightcoral'), row=2, col=1)

    # Update layout
    fig.update_layout(title_text='Distribution of Bike Rentals across Different Categorical Features', height=fig_height, width=fig_width)
    fig.update_layout(showlegend=False)

    return fig


# fig = plots.temp_vs_bike_rentals_plotly(aux_data, 1500, 600)  # Adjust width and height as needed
def temp_vs_bike_rentals_plotly(df, fig_width, fig_height):
    # Define season labels and colors
    season_details = {
        1: {'label': 'Winter', 'color': 'blue'},
        2: {'label': 'Spring', 'color': 'green'},
        3: {'label': 'Summer', 'color': 'red'},
        4: {'label': 'Fall', 'color': 'orange'}
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
        1: {'label': 'Winter', 'color': 'blue'},
        2: {'label': 'Spring', 'color': 'green'},
        3: {'label': 'Summer', 'color': 'red'},
        4: {'label': 'Fall', 'color': 'orange'}
    }

    # Create a subplot layout with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Humidity vs. Bike Rentals', 'Wind Speed vs. Bike Rentals'))

    for season, details in season_details.items():
        season_df = df[df['season'] == season]
        fig.add_trace(
            go.Scatter(x=season_df['hum'], y=season_df['cnt'],
                       mode='markers', marker=dict(size=5, color=details['color'], opacity=0.5),
                       name=details['label'], showlegend=(season == 1)),  # Show legend only for the first season
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=season_df['windspeed'], y=season_df['cnt'],
                       mode='markers', marker=dict(size=5, color=details['color'], opacity=0.5),
                       name=details['label'], showlegend=False),
            row=1, col=2
        )

    fig.update_layout(
        title_text='Humidity & Wind Speed vs. Bike Rentals',
        height=fig_height, width=fig_width
    )
    fig.update_xaxes(title_text='Normalized Humidity', row=1, col=1)
    fig.update_yaxes(title_text='Bike Rentals', row=1, col=1)
    fig.update_xaxes(title_text='Normalized Wind Speed', row=1, col=2)
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
                             line=dict(color='blue')))

    # Line plot for registered users
    fig.add_trace(go.Scatter(x=hourly_data['hr'], y=hourly_data['registered'],
                             mode='lines', name='Registered Users',
                             line=dict(color='red')))

    # Update layout
    fig.update_layout(
        title='Hourly Distribution of Casual vs. Registered Users',
        xaxis=dict(title='Hour of the Day'),
        yaxis=dict(title='Average Count'),
        width=fig_width, height=fig_height
    )

    return fig


def correlation_heatmap(df, fig_width=18, fig_height=12):
    correlation_matrix = df.corr()

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(correlation_matrix, cmap=cmap, annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap')

    # Tight layout for better spacing
    plt.tight_layout()

    return fig, correlation_matrix


def boxplot_features(df, fig_width, fig_height):
    columns_to_check = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
    data_to_plot = df[columns_to_check]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.boxplot(data=data_to_plot, orient="v", palette="Set2", width=0.5, ax=ax)
    ax.set_title('Box Plots to Identify Outliers')
    ax.set_ylabel('Value')
    ax.set_xlabel('Features')

    plt.tight_layout()
    return fig









