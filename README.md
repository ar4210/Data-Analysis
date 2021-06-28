# Data-Analysis
Collection of python and ipynb scripts used for data analysis

  # align_behavior_and_imaging.py
  - Align collected behavioral data of mouse to neuron calcium imaging data
  - Compared to original data analysis, mine uses floating point zeroes as opposed to integer zeroes.
  
  # CNMF_E.ipynb
  - Jupyter NB script that uses the CaImAn package to perform CNMFE analysis on imaging data that has already been pre processed by Inscopix software.
    - This script will be run using a secure remote server. Instructions for doing so will be included below when it has been set up. 
  
  IF YOU DO NOT HAVE ANACONDA INSTALLED:
  1. Log into ssh server using $ssh <your_uni>@<remote_server>
  2. Check if conda is installing by typing in $bash and then $conda list. If you are getting an error, it likely means you do not have anaconda installed.
  3. In order to install anaconda on the remote server, navigate to the anaconda website and copy THE LINK for the linux download.
  4. In the ssh shell, enter wget <the_copied_link> and follow the prompts.
  5. Once the set up is complete, you can check if anaconda has installed by typing in $bash followed by $conda list. You should see a list of packages installed by anaconda.

  FOR ACCESSING JUPYTER NOTEBOOKS ON YOUR REMOTE SERVER FROM YOUR LOCAL MACHINE:
  1. While logged into your server, type in the following: $ jupyter notebook --no-browser --port=8080
  2. Open a new instance of terminal, i.e. one that is connected to your machine instead of the server, and enter the following: $ ssh -N -f -L localhost:8080:localhost:8080 <your_uni>@<remote_server>
  3. Open an instance of your favorite web browser, and check the terminal window connected to the remote server for a url for Jupyter Notebooks and enter that into your window.
  4. If successful, you should be able to see the files that are stored under your user in the remote server. You should also be able to run notebooks from here, which will execute on the remote clusters.

  # CNMF_E.py
  - Same as the .ipynb, except this uses a command line argument for the file name being processed. Once the notebook is fully finished, we can use this file to run the entire CNMFE algorithm without having to run each cell individually.

  
  # spy_cnmfe.py
  - Essentially the same thing as CNMF_E.ipynb except it is a .py file. We originally ran this file in spyder IDE, where we had easy access to variables and objects that were created during the running of the script i.e. the estimates object. Also, this file includes the pickle package, which allows us to save python objects locally as files.



