import os
import shutil
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

# Repositories to clone
repos = [
    "https://huggingface.co/datasets/infinite-dataset-hub/GlobalMilestones",
    "https://huggingface.co/datasets/infinite-dataset-hub/HistoricalEvents2023",
    "https://huggingface.co/datasets/infinite-dataset-hub/GlobalCulturalShift",
    "https://huggingface.co/datasets/infinite-dataset-hub/TechAdvancements"
]

# Function to process a single repository
def process_repository(repo_url):
    folder_name = repo_url.split('/')[-1]
    data_file_path = os.path.join(folder_name, "data.csv")
    new_file_name = f"{folder_name}.csv"
    
    try:
        subprocess.run(["git", "clone", repo_url], check=True)
        if os.path.exists(data_file_path):
            shutil.move(data_file_path, new_file_name)
        shutil.rmtree(folder_name)
    except Exception as e:
        print(f"Error processing {repo_url}: {e}")

# Process each repository
for repo in repos:
    process_repository(repo)

# Check final files in the directory
print("Remaining files in the directory:", os.listdir())

# Load the datasets
try:
    df_global_milestones = pd.read_csv("GlobalMilestones.csv", index_col="idx")
    df_historical_events = pd.read_csv("HistoricalEvents2023.csv", index_col="idx")
    df_cultural_shift = pd.read_csv("GlobalCulturalShift.csv", index_col="idx")
    df_tech_advancements = pd.read_csv("TechAdvancements.csv", index_col="idx")
except FileNotFoundError as e:
    print(f"File not found: {e}")

# Add a column to identify the source of each dataset
df_global_milestones['Source'] = 'Global Milestones'
df_historical_events['Source'] = 'Historical Events 2023'
df_cultural_shift['Source'] = 'Global Cultural Shift'
df_tech_advancements['Source'] = 'Tech Advancements'

# Ensure all datasets have consistent columns (add missing columns where needed)
for df in [df_global_milestones, df_historical_events, df_cultural_shift, df_tech_advancements]:
    if 'Label' not in df.columns:
        df['Label'] = 'N/A'  # Default for datasets without the 'Label' column

# Combine the datasets
df_combined = pd.concat(
    [df_global_milestones, df_historical_events, df_cultural_shift, df_tech_advancements], 
    ignore_index=True
)

# Save combined dataset to a file
df_combined.to_csv("CombinedDataset.csv", index=False)

# Pivot data for visualization (focusing on Category distribution by Year)
category_by_year = df_combined.groupby(['Year', 'Category']).size().unstack(fill_value=0)

# Plot the stacked bar chart
fig, ax = plt.subplots(figsize=(18, 10))  # Increase figure size for better spacing
category_by_year.plot(
    kind='bar', 
    stacked=True, 
    colormap="tab20c", 
    ax=ax, 
    width=0.8  # Reduce bar width to add spacing between groups
)

# Add plot titles and labels
plt.title("Stacked Bar Plot of Category Distribution by Year", fontsize=18)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Event Count", fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(ticks=ax.get_xticks(), labels=category_by_year.index, rotation=45, ha='right')

# Adjust legend position and layout
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()  # Ensure the layout is adjusted properly for all elements

# Add gridlines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.show()

# Analyze data by 'Label' (optional)
label_counts = df_combined['Label'].value_counts()
print("Label Distribution:")
print(label_counts)


# event count visualizer


import os
import shutil
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

# Repositories to clone
repos = [
    "https://huggingface.co/datasets/infinite-dataset-hub/GlobalMilestones",
    "https://huggingface.co/datasets/infinite-dataset-hub/HistoricalEvents2023",
    "https://huggingface.co/datasets/infinite-dataset-hub/GlobalCulturalShift",
    "https://huggingface.co/datasets/infinite-dataset-hub/TechAdvancements"
    "https://huggingface.co/datasets/infinite-dataset-hub/DiscoveriesTimeline"
]

# Function to process a single repository
def process_repository(repo_url):
    folder_name = repo_url.split('/')[-1]
    data_file_path = os.path.join(folder_name, "data.csv")
    new_file_name = f"{folder_name}.csv"
    
    try:
        subprocess.run(["git", "clone", repo_url], check=True)
        if os.path.exists(data_file_path):
            shutil.move(data_file_path, new_file_name)
        shutil.rmtree(folder_name)
    except Exception as e:
        print(f"Error processing {repo_url}: {e}")

# Process each repository
for repo in repos:
    process_repository(repo)

# Check final files in the directory
print("Remaining files in the directory:", os.listdir())

# Load the datasets
try:
    df_global_milestones = pd.read_csv("GlobalMilestones.csv", index_col="idx")
    df_historical_events = pd.read_csv("HistoricalEvents2023.csv", index_col="idx")
    df_cultural_shift = pd.read_csv("GlobalCulturalShift.csv", index_col="idx")
    df_tech_advancements = pd.read_csv("TechAdvancements.csv", index_col="idx")
except FileNotFoundError as e:
    print(f"File not found: {e}")

# Add a column to identify the source of each dataset
df_global_milestones['Source'] = 'Global Milestones'
df_historical_events['Source'] = 'Historical Events 2023'
df_cultural_shift['Source'] = 'Global Cultural Shift'
df_tech_advancements['Source'] = 'Tech Advancements'

# Ensure all datasets have consistent columns (add missing columns where needed)
for df in [df_global_milestones, df_historical_events, df_cultural_shift, df_tech_advancements]:
    if 'Label' not in df.columns:
        df['Label'] = 'N/A'  # Default for datasets without the 'Label' column

# Combine the datasets
df_combined = pd.concat(
    [df_global_milestones, df_historical_events, df_cultural_shift, df_tech_advancements], 
    ignore_index=True
)

# Convert Year column to numeric for proper sorting (including BCE/negative years)
df_combined['Year'] = pd.to_numeric(df_combined['Year'], errors='coerce')
df_combined = df_combined.sort_values(by='Year')

# Save combined dataset to a file
df_combined.to_csv("CombinedDataset.csv", index=False)

# Count the number of events in each year
events_per_year = df_combined.groupby('Year').size()

# Plot the total events per year
fig, ax = plt.subplots(figsize=(20, 10))  # Larger figure size for more space
events_per_year.plot(kind='bar', ax=ax, color='skyblue', width=0.8)

# Add plot titles and labels
plt.title("Number of Events Per Year", fontsize=18)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Event Count", fontsize=14)

# Reduce the number of x-axis ticks (e.g., show every 10th year for clarity)
xticks = ax.get_xticks()
ax.set_xticks(xticks[::10])
ax.set_xticklabels(events_per_year.index[::10], rotation=45, ha='right')

# Add gridlines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()

# Analyze total event counts (optional)
print("Total Events by Year:")
print(events_per_year)
