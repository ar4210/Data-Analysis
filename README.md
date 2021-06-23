# Data-Analysis
Collection of python and ipynb scripts used for data analysis

  # align_behavior_and_imaging.py
  - Align collected behavioral data of mouse to neuron calcium imaging data
  - Compared to original data analysis, mine uses floating point zeroes as opposed to integer zeroes.
  - Data will also appear as repeating decimals rather i.e. 1.2999 vs 1.3.
  
  # CNMF_E.ipynb
  - Jupyter NB script that uses the CaImAn package to perform CNMFE analysis on imaging data that has already been pre processed by Inscopix software.
    - This script will be run using a secure remote server. Instructions for doing so will be included below when it has been set up. 
  
  # spy_cnmfe.py
  - Essentially the same thing as CNMF_E.ipynb except it is a .py file. We originally ran this file in spyder IDE, where we had easy access to variables and objects that were created during the running of the script i.e. the estimates object. Also, this file includes the pickle package, which allows us to save python objects locally as files.



