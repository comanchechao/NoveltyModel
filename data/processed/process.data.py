import os
import shutil
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

# Repositories to clone
repos = [
    "https://huggingface.co/datasets/infinite-dataset-hub/GlobalMilestones",
    "https://huggingface.co/datasets/infinite-dataset-hub/HistoricalEvents2023",
    "https://huggingface.co/datasets/infinite-dataset-hub/GlobalCulturalShift"
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
except FileNotFoundError as e:
    print(f"File not found: {e}")

# Add a column to identify the source of each dataset
df_global_milestones['Source'] = 'Global Milestones'
df_historical_events['Source'] = 'Historical Events 2023'
df_cultural_shift['Source'] = 'Global Cultural Shift'

# Combine the datasets
df_combined = pd.concat([df_global_milestones, df_historical_events, df_cultural_shift], ignore_index=True)

# Save combined dataset to a file
df_combined.to_csv("CombinedDataset.csv", index=False)

# Pivot data for visualization
category_by_year = df_combined.groupby(['Year', 'Category']).size().unstack(fill_value=0)

# Plot the stacked bar chart
category_by_year.plot(kind='bar', stacked=True, figsize=(14, 8), colormap="tab20c")
plt.title("Stacked Bar Plot of Category Distribution by Year", fontsize=18)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Event Count", fontsize=14)
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
