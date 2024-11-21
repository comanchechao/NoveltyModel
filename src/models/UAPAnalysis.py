

# Import Libraries #
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # matplotlib for plotting
import seaborn as sns  # seaborn to help with visualizations
import plotly.express as px  # plotly for interactive visualizations
import plotly.graph_objects as go  # plotly for making interactive visualizations       \

ufo_data = pd.read_csv('./data/raw/ufo.csv'),

ufo_data.head()



nulvals = ufo_data.isnull().sum()
nulpct = (nulvals / len(ufo_data))*100
print(' Null Values (% of entries):')
print(round(nulpct.sort_values(ascending=False),2))

