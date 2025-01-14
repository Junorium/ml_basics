import pandas as pd
import numpy as np

data = [] # standard dataset

### Center data
mean_data = np.mean(data)

centered_data = data - mean_data # subtract mean from original

### Standardization (z-score normalization)
mean_data = np.mean(data)
dist_data = np.std(data)

standardized_data = (data - mean_data) / dist_data # based on z-score formula

### Min-Max Normalization
max_data = max(data)
min_data = min(data)

normalized_data = (data - min_data) / (max_data - min_data) # normalized based on min, max values

### Binning
data_bins = pd.cut(data, bins=[0, 10, 20]) # bin edges

### Log Transformation
log_data = np.log(data)
