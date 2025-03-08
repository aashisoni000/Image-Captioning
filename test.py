import numpy as np
import pandas as pd
import torch


array = np.array([1, 2, 3])
print("NumPy array:", array)


data = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]})
print("Pandas DataFrame:")
print(data)


tensor = torch.tensor([1.0, 2.0, 3.0])
print("PyTorch tensor:", tensor)