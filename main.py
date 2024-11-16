# Existing code imports and setup
!pip install kaggle --upgrade
!pip install kagglehub --upgrade

import os
import kagglehub # type: ignore
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import torch # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import torch.nn as nn # type: ignore

# Download dataset
path = kagglehub.dataset_download("saketk511/world-important-events-ancient-to-modern")

# Check the downloaded files
for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))

csv_file_path = f"{path}/World Important Dates.csv"

# Load the data
data = pd.read_csv(csv_file_path)
def walk_through_dir(csv_file_path):

    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
# Extract year from the 'Date' column while ignoring month and day
data['Year'] = data['Date'].str.extract(r'(\d{4})')[0]
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

# Keep rows where the Year is known (not NaN)
data = data.dropna(subset=['Year'])

# Check how many valid entries remain
print(f"Valid entries after filtering: {len(data)}")

# Group by year to get event counts
events_per_year = data.groupby('Year').size().reset_index(name='EventCount')

# Calculate cumulative events and create Novelty Score based on cumulative count
events_per_year['CumulativeEventCount'] = events_per_year['EventCount'].cumsum()
events_per_year['NoveltyScore'] = events_per_year['CumulativeEventCount'] / events_per_year['CumulativeEventCount'].max()

# Define features and labels
X = events_per_year[['EventCount']]
y = events_per_year['NoveltyScore']

# Ensure there is data to split
if not X.empty and not y.empty:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    class EventDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.tensor(features.values, dtype=torch.float32)
            self.labels = torch.tensor(labels.values, dtype=torch.float32)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    train_dataset = EventDataset(X_train, y_train)
    test_dataset = EventDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    class NoveltyModel(nn.Module):
        def __init__(self):
            super(NoveltyModel, self).__init__()
            self.fc1 = nn.Linear(in_features=1, out_features=64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = NoveltyModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
 
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            test_loss += loss.item()
    print(f"Test Loss: {test_loss/len(test_loader)}")

    # Plotting the results
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid")

# Plot 1: Event Count per Year
    plt.subplot(3, 1, 1)
    sns.lineplot(data=events_per_year, x='Year', y='EventCount', marker="o", color="b")
    plt.title('Event Count per Year')
    plt.xlabel('Year')
    plt.ylabel('Event Count')

# Plot 2: Cumulative Event Count over Years
    plt.subplot(3, 1, 2)
    sns.lineplot(data=events_per_year, x='Year', y='CumulativeEventCount', marker="o", color="g")
    plt.title('Cumulative Event Count Over Time')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Event Count')

# Plot 3: Novelty Score per Year
    plt.subplot(3, 1, 3)
    sns.lineplot(data=events_per_year, x='Year', y='NoveltyScore', marker="o", color="r")
    plt.title('Novelty Score per Year')
    plt.xlabel('Year')
    plt.ylabel('Novelty Score (Normalized)')

# Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    events_per_year = data.groupby('Year').size().reset_index(name='EventCount')

else:
      print("Insufficient data after filtering to perform train-test split.")

