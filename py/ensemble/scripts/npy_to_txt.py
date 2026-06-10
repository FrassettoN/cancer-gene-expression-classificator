import os
import numpy as np


folders = [
    os.path.join("data", "splits", "kfold"),
    os.path.join("data", "splits", "loo")
]

for fold in folders:
    if not os.path.exists(fold):
        continue
    
    for filename in os.listdir(fold):
        if filename.endswith(".npy"):
            filepath = os.path.join(fold, filename)
            
            data = np.load(filepath, allow_pickle=True)
            
            new_name = filename.replace(".npy", ".txt")
            new_filepath = os.path.join(fold, new_name)

            np.savetxt(new_filepath, data, fmt="%s") 

