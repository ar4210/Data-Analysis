#!/usr/bin/env python
# coding: utf-8

# # Multisession registration with CaImAn
# 
# This notebook will help to demonstrate how to use CaImAn on movies recorded in multiple sessions. CaImAn has in-built functions that align movies from two or more sessions and try to recognize components that are imaged in some or all of these recordings.
# 
# The basic function for this is `caiman.base.rois.register_ROIs()`. It takes two sets of spatial components and finds components present in both using an intersection over union metric and the Hungarian algorithm for optimal matching.
# `caiman.base.rois.register_multisession()` takes a list of spatial components, aligns sessions 1 and 2, keeps the union of the matched and unmatched components to register it with session 3 and so on.


import pickle
from caiman.base.rois import register_multisession
from caiman.utils import visualization
from caiman.utils.utils import download_demo
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys


# We provide an example file generated from data courtesy of Sue Ann Koay and David Tank (Princeton University). The file contains the spatial footprints derived from the CNMF analysis of the same FOV over six different days, as well as a template (correlation image) for each day. The `download_demo` command will automatically download the file and store it in your caiman_data folder the first time you run it. To use the demo in your own dataset you can set:
# 
# ```file_path = '/path/to/file'```
# 
# or construct a list of spatial footprints and templates and use that to perform the registration as shown below.


# Load multisession data (spatial components and mean intensity templates) (should be replaced by actual data)

# file_path = download_demo('alignment.pickle')
file_path = sys.argv[1] # path to multisession file prepped by multisession_registration_prep.py
infile = open(file_path,'rb')
data = pickle.load(infile)
infile.close()

path = Path(file_path)
save_dir = Path(*(path.parts[:~0]))
leaf = Path(path.parts[~1])


'''
Lists inside of a tuple
  list1 = spatial footprints
  list2 = correlation images
  list3 = estimates objects

(
    [sp_footprint1, sp_footprint2, sp_footprint3],
    [corr_img1,     corr_img2,     corr_img3],
    [estimates1,    estimates2,    estimates3]
)
'''


print("\nPixels x Total Components")
for i in range( 1:len(data[0])+1 ):
    print(f"    Session {i}: {data[0][i].shape}")


# spatial = data[0] # list of estimates.A's for each session
# templates = data[1] # list of cn_filter's for each session
# dims = templates[0].shape # shape of the first correlation image in the pickle
# total_num_sessions = len(spatial) # total number of spatial components in pickle, i.e. how many sessions there are
# estimates_list = data[2]

spatial = data[0]
templates = data[1]
dims = templates[0].shape


print("\n")
print(f"Number of sessions: {len(data[0])}") # num of sessions
print(f"Template pixels: {data[1][0].shape}") # template pixels


# ## Use `register_multisession()`
# 
# The function `register_multisession()` requires 3 arguments:
# - `A`: A list of ndarrays or scipy.sparse.csc matrices with (# pixels X # component ROIs) for each session
# - `dims`: Dimensions of the FOV, needed to restore spatial components to a 2D image
# - `templates`: List of ndarray matrices of size `dims`, template image of each session


spatial_union, assignments, matchings = register_multisession(A=spatial, dims=dims, templates=templates)


# The function returns 3 variables for further analysis:
# - `spatial_union`: csc_matrix (# pixels X # total distinct components), the union of all ROIs across all sessions aligned to the FOV of the last session.
# - `assignments`: ndarray (# total distinct components X # sessions). `assignments[i,j]=k` means that component `k` from session `j` has been identified as component `i` from the union of all components, otherwise it takes a `NaN` value. Note that for each `i` there is at least one session index `j` where `assignments[i,j]!=NaN`.
# - `matchings`: list of (# sessions) lists. Saves `spatial_union` indices of individual components in each session. `matchings[j][k] = i` means that component `k` from session `j` is represented by component `i` in the union of all components `spatial_union`. In other words `assignments[matchings[j][k], j] = j`.

# ## Post-alignment screening

# The three outputs can be used to filter components in various ways. For example we can find the components that were active in at least a given a number of sessions. For more examples, check [this script](https://github.com/flatironinstitute/CaImAn/blob/master/use_cases/eLife_scripts/figure_9/Figure_9_alignment.py) that reproduces the results of [Figure 9, as presented in our eLife paper](https://elifesciences.org/articles/38173#fig9).


# Filter components by number of sessions the component could be found
n_reg = 8 # minimal number of sessions that each component has to be registered in

# Use number of non-NaNs in each row to filter out components that were not registered in enough sessions
assignments_filtered = np.array(np.nan_to_num(assignments[np.sum(~np.isnan(assignments), axis=1) >= n_reg]), dtype=int);
# Use filtered indices to select the corresponding spatial components

spatial_filtered = spatial[0][:, assignments_filtered[:,0]]
    

# Plot spatial components of the selected components on the template of the last session

coordinates = visualization.plot_contours(spatial_filtered, templates[-1]);
print(spatial_filtered.shape)

coordinates_df = pd.DataFrame(coordinates)
coordinates_df.to_csv(f"{save_dir}/coordinates_{leaf}.csv")


# ## Combining data of components over multiple sessions
# 
# Now that all sessions are aligned and we have a list of re-registered neurons, we can use `assignments` and `matchings` to collect traces from neurons over different sessions.
# 
# As an exercise, we can collect the traces of all neurons that were registered in all sessions. We already gathered the indices of these neurons in the previous cell in `assignments_filtered`. Assuming that traces of each session are saved in their own `CNMF` object collected in a list, we can iterate through `assignments_filtered` and use these indices to find the re-registered neurons in every session.
# 
# Note: This notebook does not include the traces of the extracted neurons, only their spatial components. As such the loop below will produce an error. However, it demonstrates how to use the results of the registration to in your own analysis to extract the traces of the same neurons across different sessions.


print(spatial_filtered.shape)


# Now we have the array `traces`, where element `traces[i,j] = k` is the temporal component of neuron `i` at session `j`. This can be performed with `F_dff` data or `S` spikes as well.




