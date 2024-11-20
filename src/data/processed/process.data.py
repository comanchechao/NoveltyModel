import os
import shutil

# List of repositories to clone
repos = [
    "https://huggingface.co/datasets/infinite-dataset-hub/GlobalMilestones",
    "https://huggingface.co/datasets/infinite-dataset-hub/HistoricalEvents2023",
    "https://huggingface.co/datasets/infinite-dataset-hub/GlobalCulturalShift"
]

# Cloning repositories and processing files
for repo in repos:
    # Extract folder name from the repo URL
    folder_name = repo.split('/')[-1]
    
    # Clone the repository
    os.system(f"git clone {repo}")
    
    # Path to the data.csv file in the cloned directory
    data_file_path = os.path.join(folder_name, "data.csv")
    
    # New name for the file
    new_file_name = f"{folder_name}.csv"
    
    # Move the file to the current directory with the new name
    if os.path.exists(data_file_path):
        shutil.move(data_file_path, new_file_name)
    
    # Remove the cloned directory
    shutil.rmtree(folder_name)

# Check the final files in the directory
print("Remaining files in the directory:", os.listdir())


import pandas as pd

# Load the datasets
df_global_milestones = pd.read_csv("GlobalMilestones.csv", index_col="idx")
df_historical_events = pd.read_csv("HistoricalEvents2023.csv", index_col="idx")
df_cultural_shift = pd.read_csv("GlobalCulturalShift.csv", index_col="idx")

# Add a column to identify the source of each dataset
df_global_milestones['Source'] = 'Global Milestones'
df_historical_events['Source'] = 'Historical Events 2023'
df_cultural_shift['Source'] = 'Global Cultural Shift'

# Combine the datasets into a single DataFrame
df_combined = pd.concat([df_global_milestones, df_historical_events, df_cultural_shift], ignore_index=True)

# Display the last few rows of the combined DataFrame
print(df_combined.head())


# visualize data

import matplotlib.pyplot as plt

# Pivot the data to get the counts of each category by year
category_by_year = df_combined.groupby(['Year', 'Category']).size().unstack(fill_value=0)

# Plot the stacked bar chart
category_by_year.plot(kind='bar', stacked=True, figsize=(14, 8), colormap="viridis")

# Add plot titles and labels
plt.title("Stacked Bar Plot of Category Distribution by Year", fontsize=18)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Event Count", fontsize=14)

# Adjust legend position and layout
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Display the plot
plt.show()
