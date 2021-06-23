# -*- coding: utf-8 -*-
#%%
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.source_extraction.cnmf import params as params

import os
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

#%%
print(os.getcwd())


#%% CD into Movies folder
print("Current dir: ", os.getcwd(), "\nChanging dir...")
os.chdir("C:\\Users\\wmf2107\\caiman_data\\Aditya\\Movies")
print("CWD changed to: ", os.getcwd())


#%% Stop cluster if one already exists. Start a new cluster.
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
                backend='local', 
                n_processes=24, # number of process to use, if you go out of memory try to reduce this one
                single_thread=False)


#%% NAME YOUR FILE(S) TO BE PROCESSED
f = ["crop_MC_recording_20161015_162122.tif"]
# f = ["full_moco.tif"]
f_mm = cm.save_memmap(f, base_name='memmap_',order='C', border_to_0 = 0, dview=dview)


#%%
Yr, dims, T = cm.load_memmap(f_mm)
images = Yr.T.reshape((T,) + dims, order='F')


#%%
# parameters for source extraction and deconvolution
p = 1               # order of the autoregressive system
K = None            # upper bound on number of components per patch, in general None
gSig = (3, 3)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = (13, 13)     # average diameter of a neuron, in general 4*gSig+1
Ain = None          # possibility to seed with predetermined binary masks
merge_thr = .7      # merging threshold, max correlation allowed
rf = 40             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
stride_cnmf = 20    # amount of overlap between the patches in pixels
#                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
tsub = 2            # downsampling factor in time for initialization,
#                     increase if you have memory problems
ssub = 1            # downsampling factor in space for initialization,
#                     increase if you have memory problems
#                     you can pass them here as boolean vectors
low_rank_background = None  # None leaves background of each patch intact,
#                     True performs global low-rank approximation if gnb>0
gnb = 0             # number of background components (rank) if positive,
#                     else exact ring model with following settings
#                         gnb= 0: Return background as b and W
#                         gnb=-1: Return full rank background B
#                         gnb<-1: Don't return background
nb_patch = 0        # number of background components (rank) per patch if gnb>0,
#                     else it is set automatically
min_corr = .8       # min peak value from correlation image
min_pnr = 10        # min peak to noise ration from PNR image
ssub_B = 2          # additional downsampling factor in space for background
ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor


opts = params.CNMFParams(params_dict={
                                'dims': dims,
                                'method_init': 'corr_pnr',  # use this for 1 photon
                                'K': K,
                                'gSig': gSig,
                                'gSiz': gSiz,
                                'merge_thr': merge_thr,
                                'p': p,
                                'tsub': tsub,
                                'ssub': ssub,
                                'rf': rf,
                                'stride': stride_cnmf,
                                'only_init': True,    # set it to True to run CNMF-E
                                'nb': gnb,
                                'nb_patch': nb_patch,
                                'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                'low_rank_background': low_rank_background,
                                'update_background_components': True,  # sometimes setting to False improve the results
                                'min_corr': min_corr,
                                'min_pnr': min_pnr,
                                'normalize_init': False,               # just leave as is
                                'center_psf': True,                    # leave as is for 1 photon
                                'ssub_B': ssub_B,
                                'ring_size_factor': ring_size_factor,
                                'del_duplicates': True,                # whether to remove duplicates from initialization
                                'border_pix': 0})                # number of pixels to not consider in the borders)



#%% 
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::5], gSig=gSig[0], swap_dim=False)
inspect_correlation_pnr(cn_filter, pnr)
print(min_corr)
print(min_pnr)


#%% RUN CNMF ON PATCHES IN PARALLEL
### For a (.tiff) file size of 5.9 GB, this took about 1.5 hrs to complete ###

cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, params=opts)
cnm = cnm.fit(images)



#%% COMPONENT EVALUATION - FROM ORIGINAL NOTEBOOK IN GITHUB
# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier

min_SNR = 2.5          # adaptive way to set threshold on the transient size
r_values_min = 0.85    # threshold on space consistency (if you lower more components
#                        will be accepted, potentially with worst quality)
cnm.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})
estimates_object = cnm.estimates.evaluate_components(images, cnm.params, dview=None)

print(' ***** ')
print('Number of total components: ', len(estimates_object.C))
print('Number of accepted components: ', len(estimates_object.idx_components))




#%%
cnm.dims = dims
display_images = True
if display_images:
    cnm.estimates.plot_contours(img = cn_filter, idx = cnm.estimates.idx_components)
    cnm.estimates.view_components(images, idx=cnm.estimates.idx_components)

#%% Extract Data to CSV


tmp = f[0].index('.')
save_as = f'{f[0][:tmp]}'

if os.path.isdir(f"C:\\Users\\wmf2107\\caiman_data\\Aditya\\Data\\{save_as}"):
    print("Directory Already Exists")
else:
    os.mkdir(f"C:\\Users\\wmf2107\\caiman_data\\Aditya\\Data\\{save_as}")

# Inferred Spikes
estimates_spikes = pd.DataFrame(estimates_object.S)
estimates_spikes.to_csv(path_or_buf = f"C:\\Users\\wmf2107\\caiman_data\\Aditya\\Data\\{save_as}\\spikes.csv", 
                                 index_label = "Cell Number", 
                                 header = ["fr " + str(i) for i in range(estimates_cell_extraction.shape[1])])

# Calcium Traces
estimates_C = pd.DataFrame(estimates_object.C)
estimates_C.to_csv(path_or_buf = f"C:\\Users\\wmf2107\\caiman_data\\Aditya\\Data\\{save_as}\\ca_traces.csv",
                   index_label = "Cell Number",
                   header = ["fr " + str(i) for i in range(estimates_cell_extraction.shape[1])])

# Neuron Coordinates / bounding boxes
estimates_coord = pd.DataFrame(estimates_object.coordinates)
estimates_coord.to_csv(path_or_buf = f"C:\\Users\\wmf2107\\caiman_data\\Aditya\\Data\\{save_as}\\coordinates.csv")


#%% To print/access one column of a DataFrame
print(estimates_coord["CoM"])

#%%
import pickle

with open('estimates.pickle', 'wb') as estimates_object_file:
    pickle.dump(estimates_object, estimates_object_file)
    
#%%

with open('estimates.pickle', 'rb') as estimates_object_file:
    estimates_object_2 = pickle.load(estimates_object_file)