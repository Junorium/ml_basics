import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler() # takes in an array from [-1,1]
mmscaler = MinMaxScaler()

data = [] # standard dataset
data_reshaped = np.array(data).reshape(-1,1) # normalize from -1 to 1

### Standardization (z-score normalization)
standardized_data = scaler.fit_transform(data_reshaped)

### Min-Max Normalization
normalized_data = mmscaler.fit_transform(data_reshaped)


