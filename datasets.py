from torch.utils.data import DataLoader, Dataset
from netCDF4 import Dataset as ncDataset
from datetime import datetime
import os
import pandas as pd
import numpy as np
import torch

class FloodingDatasetStack(Dataset):
    def __init__(self, nc_dir, label_dir, cities, stack_size = 5, start_year=1993, end_year=2013):
        self.nc_dir = nc_dir
        self.cities = cities
        self.stack_size = stack_size
        self.data = []
        self.labels = []
        self._prepare_data(label_dir, start_year, end_year)

    def _prepare_data(self, label_dir, start_year, end_year):
        # Load anomaly data from CSV files
        cities = [f for f in os.listdir(label_dir) if f.endswith('.csv')]
        anomaly_data = []
        for file in cities:
            file_path = os.path.join(label_dir, file)
            df = pd.read_csv(file_path)
            anomaly_data.append(df['anomaly'].iloc[:7064].values)

        self.labels = np.array(anomaly_data).T

        # Load .nc files
        ncfiles = sorted([f for f in os.listdir(self.nc_dir) if f.endswith('.nc')])  # Ensure sorted order by date
        for file in ncfiles:
            file_path = os.path.join(self.nc_dir, file)
            self.data.append(file_path)
            if len(self.data) == 7064:  # Ensure the dataset length matches the label length
                break

    def __len__(self):
        # Ensure the dataset length accounts for the window size
        return len(self.data) - self.stack_size + 1

    def __getitem__(self, idx):
        # Load a stack of 5 consecutive time steps
        nc_files = self.data[idx:idx + self.stack_size]
        sea_levels = []

        for nc_file in nc_files:
            nc_data = ncDataset(nc_file, 'r')
            sea_level = np.array(np.squeeze(nc_data.variables['sla'][:]))  # Adjust for your variable name
            nc_data.close()
            sea_levels.append(torch.tensor(sea_level, dtype=torch.float32).unsqueeze(0))  # Add channel dimension

        # Stack the sea levels along the channel dimension
        sea_levels = torch.cat(sea_levels, dim=0)  # Shape: 5x100x160

        # Get the label for the current time step
        label = torch.tensor(self.labels[idx + self.stack_size - 1], dtype=torch.float32)

        return sea_levels, label


# Custom Dataset for Flooding Data
class FloodingDataset(Dataset):
    def __init__(self, nc_dir, label_dir, cities, start_year=1993, end_year=2013):
        self.nc_dir = nc_dir
        self.cities = cities
        self.data = []
        self.labels = []
        self._prepare_data(label_dir, start_year, end_year)

    def _prepare_data(self, label_dir, start_year, end_year):
        cities = [f for f in os.listdir(label_dir) if f.endswith('.csv')]
        anomaly_data = []
        for file in cities:
            file_path = os.path.join(label_dir, file)
            df = pd.read_csv(file_path)
            anomaly_data.append(df['anomaly'].iloc[:7064].values)

        self.labels = np.array(anomaly_data).T

        ncfiles = [f for f in os.listdir(self.nc_dir) if f.endswith('.nc')]

        for file in ncfiles:
            file_path = os.path.join(self.nc_dir, file)
            self.data.append(file_path)
            if len(self.data) == 7064: break

        # for city in self.cities:
        #     # Load the city-specific CSV
        #     label_path = os.path.join(label_dir, f"{city.replace(' ', '_')}_{start_year}_{end_year}_training_data.csv")
        #     labels = pd.read_csv(label_path)
            
        #     # Iterate through the dates and match with .nc files
        #     for _, row in labels.iterrows():
        #         date = datetime.strptime(row['t'], "%Y-%m-%d")
        #         nc_file = os.path.join(
        #             self.nc_dir,
        #             f"dt_ena_{date.strftime('%Y%m%d')}_vDT2021.nc"
        #         )
        #         if os.path.exists(nc_file):
        #             if city == 'Atlantic City':
        #                 self.data.append(nc_file)
        #             # self.labels.append(row['anomaly'])  # 0 or 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load .nc file and anomaly label
        nc_file = self.data[idx]
        nc_data = ncDataset(nc_file, 'r')
        sea_level = np.array(np.squeeze(nc_data.variables['sla'][:]))  # Adjust for your variable name
        nc_data.close()
        sea_level = torch.tensor(sea_level, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sea_level, label
