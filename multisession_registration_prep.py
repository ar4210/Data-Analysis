#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle as p
import os
import sys
from pathlib import Path
import datetime
import caiman


def prep_multisession_registration():
    '''
    Command line argument example:
        python 'ar4210_data_analysis/CaImAn-master/aditya/multisession_registration_prep.py' 
        '/home/ar4210/engram/Mouse/New_Analysis_Pipeline_Test/Test_Data/Python_Scripts_and_Data/Pickles/wfC321_test'
        
    Returns:
      Pickle File containing tuple of lists of spatial footprints, correlation image, and estimates objects for each recording
        
      Lists inside of a tuple:
          list1 = spatial footprints
          list2 = correlation images
          list3 = estimates objects

        (
            [sp_footprint1, sp_footprint2, sp_footprint(n)],
            [corr_img1,     corr_img2,     corr_img(n)],
            [estimates1,    estimates2,    estimates(n)]
        )

    '''
    mouse_id_dir = sys.argv[1]
    print(f"\n{mouse_id_dir}")
    assert len(sys.argv) == 2, "Please provide path to mouse recordings."
    assert os.path.exists(mouse_id_dir), "MouseID directory does not exist. Check spelling and location."
    print(f"MouseID directory {os.path.basename(mouse_id_dir)} found.")

    path = Path(mouse_id_dir)
    mouse_id = path.stem # Takes the last item in the path provided. "home_dir/next_dir/file.py" --> "file.py"


    sp_footprints_master = []
    corr_img_master = []
    estimates_master = []


    print(f"Iterating through {mouse_id} recordings... Extracting Spatial Footprints, Correlation Images, and Estimates Objects.\n")
    
    for i in os.listdir(mouse_id_dir):
        if os.path.exists(f"{mouse_id_dir}/{i}") and os.path.isdir(f"{mouse_id_dir}/{i}"):
            print(f"\033[96mRecording ID directory {i} found. Extracting Data...\033[0m")
            for j in os.listdir(f"{mouse_id_dir}/{i}"):
                if os.path.exists(f"{mouse_id_dir}/{i}/{j}") and os.path.isdir(f"{mouse_id_dir}/{i}/{j}"):
                    for k in os.listdir(f"{mouse_id_dir}/{i}/{j}"):
                        if 'spatial_footprints' in k:
                            with open(f"{mouse_id_dir}/{i}/{j}/{k}", "rb") as sp_footprints:
                                spf = p.load(sp_footprints)
                            sp_footprints_master.append(spf)
                        elif 'correlation_img' in k:
                            with open(f"{mouse_id_dir}/{i}/{j}/{k}", "rb") as correlation_img:
                                corr_img = p.load(correlation_img)
                            corr_img_master.append(corr_img)
                        elif 'estimates' in k:
                            with open(f"{mouse_id_dir}/{i}/{j}/{k}", "rb") as estimates:
                                estims = p.load(estimates)
                            estimates_master.append(estims)
                            
    print(f"sp: {len(sp_footprints_master)}\nci: {len(corr_img_master)}\nem: {len(estimates_master)}")
    # Check to make sure footprints and images were added to lists
    if len(sp_footprints_master) > 0 and len(corr_img_master) > 0: # and len(estimates_master) > 0:
        now = datetime.datetime.now()
        ctime = now.strftime("%Y%m%d_%H%M%S")

        footprints_ci_estimates = (sp_footprints_master, corr_img_master, estimates_master)
        with open(f"{mouse_id_dir}/{mouse_id}_{ctime}_multisession_registration.pkl", "wb") as tmp:
            p.dump(footprints_ci_estimates, tmp)

    print(f"\033[92mDone! Multisession Registration file saved in location:\n\033[0m{mouse_id_dir}/{mouse_id}_{ctime}_multisession_registration.pkl")
    
if __name__ == "__main__":
    prep_multisession_registration()
        
    

