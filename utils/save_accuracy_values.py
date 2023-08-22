import numpy as np
import os
#save closed oa and open oa values per epoch
def save_accuracy_values(closed_oa, open_oa, filepath):
    """
    Saves closed and open overall accuracy scalar values per epoch to a numpy file.

    Args:
        closed_oa: A scalar value representing the closed overall accuracy per epoch.
        open_oa: A scalar value representing the open overall accuracy per epoch.
        epoch: An integer representing the epoch number.
        filepath: A string representing the file path to save the accuracy values to.
    """
    data = np.array([[closed_oa, open_oa]])

    # Check if the file exists
    if os.path.isfile(filepath):
        # Load the existing numpy file
        npfile = np.load(filepath)
        # Append the new data to the numpy file
        npfile = np.concatenate((npfile, data), axis=0)
        # Save the modified numpy file
        np.save(filepath, npfile)
    else:
        # Create a new numpy file with the data
        np.save(filepath, data)

    print(f"Accuracy values saved to {filepath}")