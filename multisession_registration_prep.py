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
    mouse_id_dir = sys.argv[1]


    if len(sys.argv) < 2:
        print("Please provide path to mouse recordings.")
        quit()

    if not os.path.exists(mouse_id_dir):
        print("\033[91mMouseID directory does not exist. Check spelling and location.\033[0m")
        quit()
    else:
        print(f"\033[92mMouseID directory {os.path.basename(mouse_id_dir)} found. Moving on...\033[0m")


    path = Path(mouse_id_dir)
    mouse_id = path.stem # Takes the last item in the path provided. "home_dir/next_dir/file.py" --> "file.py"


    sp_footprints_master = []
    corr_img_master = []
    estimates_master = []


    print(f"Iterating through {mouse_id} recordings... Extracting Spatial Footprints and Correlation Images.\n")
    
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
                        elif 'ESTIMATES' in k:
                            with open(f"{mouse_id_dir}/{i}/{j}/{k}", "rb") as estimates:
                                estims = p.load(estimates)
                            estimates_master.append(estims)
    # Check to make sure footprints and images were added to lists
    if len(sp_footprints_master) > 0 and len(corr_img_master) > 0 and len(estimates_master) > 0:
        now = datetime.datetime.now()
        ctime = now.strftime("%Y%m%d_%H%M%S")

        footprints_ci_estimates = (sp_footprints_master, corr_img_master, estimates_master)
        with open(f"{mouse_id_dir}/{mouse_id}_{ctime}_multisession_registration.pkl", "wb") as tmp:
            p.dump(footprints_ci_estimates, tmp)

    print(f"\033[92mDone! Multisession Registration file saved in location:\n\033[0m{mouse_id_dir}/{mouse_id}_{ctime}_multisession_registration.pkl")
    
if __name__ == "__main__":
    prep_multisession_registration()
        
    

