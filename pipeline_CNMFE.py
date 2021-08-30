#!/usr/bin/env python

"""
Complete pipeline for motion correction, source extraction, and deconvolution
of one photon microendoscopic calcium imaging data using the CaImAn package.
The demo demonstrates how to use the params, MotionCorrect and cnmf objects
for processing 1p microendoscopic data. The analysis pipeline is similar as in
the case of 2p data processing with core difference being the usage of the
CNMF-E algorithm for source extraction (as opposed to plain CNMF). Check
the companion paper for more details.

You can also run a large part of the pipeline with a single method
(cnmf.fit_file) See inside for details.

Demo is also available as a jupyter notebook (see demo_pipeline_cnmfE.ipynb)
"""

import os
import sys

# Check if files were passed in at the command line. If no files, quit
if ( len(sys.argv) < 2 ):
    print("\033[91mERROR: \033[0mCommand Line Arguments missing.")
    quit()
    
# Print command line argument files to process. If files could not be found or don't exist, quit
print("\nFiles to be processed ...")    
for index, file in enumerate(sys.argv[1:], 1):
    print(f"{index} .......... {file}")
    if not ( os.path.exists(file) ):
        print("\033[91mFile does not exist. Check spelling and location.\033[0m")
        quit()
    else:
        print(f"\033[92mFile {os.path.basename(file)} found. Moving on...\033[0m")
    
new_dir = "engram/anole/Mouse/New_Analysis_Pipeline_Test/Test_Data/Python_Scripts_and_Data" 
os.chdir(new_dir)
print(f"\nChanged working directory to: \n{new_dir}\n")


import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import time
import datetime
import pandas as pd

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params

#%%
# Set up the logger; change this if you like.
# You can log to a file using the filename parameter, or make the output more or less
# verbose by setting level to logging.DEBUG, logging.INFO, logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "[%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "%(message)s",
#                     "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
#                     "[%(process)d] %(message)s",
                    level=logging.WARNING)
    # filename="/tmp/caiman.log"
    
    
base = os.getcwd() # Parent dir of raw videos dir and pickles dir
data = os.path.join(base, "RawFiles") # Dir where raw video files (.hdf5) will be stored
pickles = os.path.join(base, "Pickles") # Dir where pickle files (cnm.estimates objects) will be stored

fnames = [file]  # filename to be processed

path = Path(file)
mouse_id = path.parts[-2] # Store pickle files / processed data in a folder with this name
data_dir = Path(*(path.parts[:-1])) # path to file without file name included

fstem = path.stem # name without extension
extension = path.suffix # .tif, .hdf5, etc.
ffull = path.name # fstem + extension
    
    
def main(file):
    
    start = time.time()
      
        
    base = os.getcwd() # Parent dir of raw videos dir and pickles dir
    data = os.path.join(base, "RawFiles") # Dir where raw video files (.hdf5) will be stored
    pickles = os.path.join(base, "Pickles") # Dir where pickle files (cnm.estimates objects) will be stored

    fnames = [file]  # filename to be processed

    path = Path(file)
    mouse_id = path.parts[-2] # Store pickle files / processed data in a folder with this name
    data_dir = Path(*(path.parts[:-1])) # path to file without file name included

    fstem = path.stem # name without extension
    extension = path.suffix # .tif, .hdf5, etc.
    ffull = path.name # fstem + extension

    
    # %% start the cluster
    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=None,  # Nmbr of process to use, reduce if you run out of memory
                                                     single_thread=False)
    
    # %% First setup some parameters for motion correction
    # dataset dependent parameters
    
    fr = 20                          # movie frame rate
    decay_time = 0.4                 # length of a typical transient in seconds

    
    # motion correction parameters
    motion_correct = True            # flag for motion correction
    pw_rigid = False                 # flag for pw-rigid motion correction

    
    gSig_filt = (3, 3)   # size of filter, in general gSig (see below), change this one if algorithm does not work
    max_shifts = (5, 5)  # maximum allowed rigid shift
    strides = (48, 48)   # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
                             # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3
    border_nan = 'copy'

    mc_dict = {
        'fnames': fnames,
        'fr': fr,
        'decay_time': decay_time,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan,
        'var_name_hdf5':'images'
    }

    opts = params.CNMFParams(params_dict=mc_dict)

# %% MOTION CORRECTION
#  The pw_rigid flag set above, determines where to use rigid or pw-rigid
#  motion correction
    if motion_correct:
        # do motion correction rigid
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'), var_name_hdf5 = 'images')
        mc.motion_correct(save_movie=True)
        fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
        if pw_rigid:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                         np.max(np.abs(mc.y_shifts_els)))).astype(int)
        else:
            bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
#             plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
#             plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
#             plt.legend(['x shifts', 'y shifts'])
#             plt.xlabel('frames')
#             plt.ylabel('pixels')

        bord_px = 0 if border_nan == 'copy' else bord_px
        fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                                   border_to_0=bord_px)
    else:  # if no motion correction just memory map the file
        fname_new = cm.save_memmap(filename_reorder, base_name='memmap_',
                                   order='C', border_to_0=0, dview=dview)

    # load memory mappable file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')

# %% Parameters for source extraction and deconvolution (CNMF-E algorithm)

    p = 1               # order of the autoregressive system
    K = None            # upper bound on number of components per patch, in general None for 1p data
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

    opts.change_params(params_dict={'dims': dims,
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
                                    'border_pix': bord_px})                # number of pixels to not consider in the borders)

# %% compute some summary images (correlation and peak to noise)
    # change swap dim if output looks weird, it is a problem with tiffile
    cn_filter, pnr = cm.summary_images.correlation_pnr(images[::5], gSig=gSig[0], swap_dim=False)
    # if your images file is too long this computation will take unnecessarily
    # long time and consume a lot of memory. Consider changing images[::1] to
    # images[::5] or something similar to compute on a subset of the data

    # inspect the summary images and set the parameters
    inspect_correlation_pnr(cn_filter, pnr)
    # parameters set above, modify them if necessary based on summary images
#     print(min_corr) # min correlation of peak (from correlation image)
#     print(min_pnr)  # min peak to noise ratio

# %% RUN CNMF ON PATCHES
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
    cnm.fit(images)

# %% ALTERNATE WAY TO RUN THE PIPELINE AT ONCE
    #   you can also perform the motion correction plus cnmf fitting steps
    #   simultaneously after defining your parameters object using
#    cnm1 = cnmf.CNMF(n_processes, params=opts, dview=dview)
#    cnm1.fit_file(motion_correct=True)

# %% DISCARD LOW QUALITY COMPONENTS
    min_SNR = 2.5           # adaptive way to set threshold on the transient size
    r_values_min = 0.95    # threshold on space consistency (if you lower more components
    #                        will be accepted, potentially with worst quality)
    cnm.params.set('quality', {'min_SNR': min_SNR,
                               'rval_thr': r_values_min,
                               'use_cnn': True})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    
    print('\n')
    print(' ***** ')
    print(f"Number of total components: \033[92m{len(cnm.estimates.C)}\033[0m")
    print(f"Number of accepted components: \033[92m{len(cnm.estimates.idx_components)}\033[0m")
    print(' ***** ')
    print('\n')
    
# # %% PLOT COMPONENTS
#     cnm.dims = dims
#     display_images = True           # Set to true to show movies and images
#     if display_images:
#         cnm.estimates.plot_contours(img=cn_filter, idx=cnm.estimates.idx_components)
#         cnm.estimates.view_components(images, idx=cnm.estimates.idx_components)


# %% STOP SERVER
    cm.stop_server(dview=dview)
    
    
# %% SAVE FILES
    now = datetime.datetime.now()
    ctime = now.strftime("%Y%m%d_%H%M%S")
        
    os.makedirs(f"{pickles}/{mouse_id}/{fstem}", exist_ok = True)

    with open(f"{pickles}/{mouse_id}/{fstem}/{fstem}_ESTIMATES_{ctime}.pkl", "wb") as tmp:
        pickle.dump(cnm.estimates, tmp)
    
    print(f"{fstem} estimates object saved: \033[1m{pickles}/{mouse_id}/{fstem}/{fstem}_ESTIMATES_{ctime}.pkl\033[0m \n")
        
    spatial_footprints = pd.DataFrame(cnm.estimates.A)
    inferred_spikes = pd.DataFrame(cnm.estimates.S)
    calcium_traces = pd.DataFrame(cnm.estimates.C)
    contours = pd.DataFrame(cnm.estimates.coordinates)

    spatial_footprints.to_csv(f"{pickles}/{mouse_id}/{fstem}/{fstem}_SP_FOOTPRINTS_{ctime}.csv")
    inferred_spikes.to_csv(f"{pickles}/{mouse_id}/{fstem}/{fstem}_SPIKES_{ctime}.csv")
    calcium_traces.to_csv(f"{pickles}/{mouse_id}/{fstem}/{fstem}_TRACES_{ctime}.csv")
    contours.to_csv(f"{pickles}/{mouse_id}/{fstem}/{fstem}_CONTOURS_{ctime}.csv")
    
    end = time.time()
    prettify_time(start, end, ffull)

    
    return cnm.estimates.A, cn_filter, mouse_id


    
def prettify_time(start, end, file):
    ''' 
    Takes elapsed time in seconds and converts to a readable format, HH:MM:SS. Prints to terminal.
    '''
    print('\n*****')
    print(f"Elapsed time for { file } processing (HH:MM:SS): \033[92m{ str(datetime.timedelta(seconds = (end - start))) }\033[0m")
    print('*****\n')
    
    
def cleanup_memmaps(path):
    '''
    Takes in path to where mmaps are saved, and deletes them.
    '''
    for i in (os.listdir(path)):
        print(i)
    print('\n\n')
    for filename in os.listdir(path):
        if filename.endswith(".mmap"):
#             os.remove(filename)
            print(filename)
            os.remove(filename)
    print("\n\n")
    for i in (os.listdir(path)):
        print(i)
            
          
    
# %% This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    full_start = time.time()
    spatial_footprints_list = []
    corr_image_list = []
    for i in range(1,len(sys.argv)):
        print(sys.argv[i])
        spatial_footprints, corr_image, mouse_id = main(sys.argv[i])
        spatial_footprints_list.append(spatial_footprints)
        corr_image_list.append(corr_image)
    full_end = time.time()
    prettify_time(full_start, full_end, os.path.basename(__file__))
    with open(f"{pickles}/{mouse_id}/{mouse_id}_multisession_registration.pkl", "wb") as multi_pkl:
        pickle.dump(spatial_footprints_list, multi_pkl)
        pickle.dump(corr_image_list, multi_pkl)
    print(len(spatial_footprints_list), spatial_footprints_list)
    print(len(corr_image_list), corr_image_list)
        