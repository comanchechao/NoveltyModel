import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # matplotlib for plotting
import seaborn as sns  # seaborn to help with visualizations

ufo_data = pd.read_csv('./data/raw/ufo.csv', 
                        low_memory = False, 
                        na_values = ['UNKNOWN','UNK'], 
                        na_filter = True, 
                        skip_blank_lines = True)
ufo_data.head()





ufo_subcols = ['datetime', 'city', 'state', 'country', 'shape', 'duration (seconds)',
        'comments', 'date posted', 'latitude',
       'longitude ']

ufo_data = pd.DataFrame(data=ufo_data, columns=ufo_subcols)

ufo_data = ufo_data.dropna(thresh=8)

ufo_data = ufo_data.reset_index(drop=True)

ufo_data['latitude'] = pd.to_numeric(ufo_data['latitude'],errors = 'coerce')  # latitudes as numerics
ufo_data['longitude '] = pd.to_numeric(ufo_data['longitude '], errors='coerce')

ufo_date = ufo_data.datetime.str.replace('24:00', '00:00')  # clean illegal values
ufo_date = pd.to_datetime(ufo_date, format='%m/%d/%Y %H:%M')  # now in datetime

ufo_data['datetime'] = ufo_data.datetime.str.replace('24:00', '00:00')
ufo_data['datetime'] = pd.to_datetime(ufo_data['datetime'], format='%m/%d/%Y %H:%M')
ufo_year = ufo_data['datetime'].dt.year
years_data = ufo_year.value_counts()
years_index = years_data.index
years_values = years_data.get_values()

custom_palette = ['#F0386B']

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_facecolor('#0E0E0D')
fig.patch.set_facecolor('#1A1919')  
plt.xticks(rotation=45 , color = 'white')
plt.yticks(rotation=45 , color = 'white')
plt.title('UFO Sightings by Year', color = 'white')
plt.bar(years_index, years_values, color = 'skyblue')
plt.ylabel("Sightings", color="white")
plt.xlabel("Years", color="white")
plt.show
years_plot = sns.barplot(x=years_index[:60],y=years_values[:60], palette = custom_palette)