# Data-Analysis
Collection of python and ipynb scripts used for data analysis

Prerequisites:
- Your own virtual machine (VM) using the Cortex Clusters in the Zuckerman Institute. If you are NOT using your own VM, skip the steps about accessing your remote server.
- I did this on a 2015 macbook pro. Other macbook versions and linux OSes are probably supported.

IF YOU DO NOT HAVE ANACONDA INSTALLED ON YOUR VM:
  1. Log into your VM using $ssh <your_uni>@<remote_server>
  2. Check if conda is installing by typing in $bash and then $conda list. If you are getting an error, it likely means you do not have anaconda installed.
  3. In order to install anaconda on the remote server, navigate to the anaconda website and copy THE LINK for the linux download.
  4. In the ssh shell, enter wget <the_copied_link> and follow the prompts.
  5. Once the set up is complete, you can check if anaconda has installed by typing in $bash followed by $conda list. You should see a list of packages installed by anaconda.

  FOR ACCESSING JUPYTER NOTEBOOKS ON YOUR REMOTE SERVER FROM YOUR LOCAL MACHINE:
  0. If on Windows machine, follow tutorial [here](https://itsfoss.com/install-bash-on-windows/) to use linux terminal on Windows. I did this on my macbook (and I think any linux OS should be similar), but for Windows you will likely have to adapt each step for your OS / the windows terminal.
  1. While logged into your server, type in the following: $ jupyter notebook --no-browser --port=8080 --NotebookApp.iopub_data_rate_limit=1.0e10
  2. Open a new instance of terminal, i.e. one that is connected to your machine instead of the server, and enter the following: $ ssh -N -f -L localhost:8080:localhost:8080 <your_uni>@<remote_server>
      1. If you receive an error message saying the port is in use, or are otherwise unable to access it, try running $ pgrep ssh and $ killall ssh. This will log you out of all ssh entry points and you will have to log back in, but port 8080 should now be open.
  4. Open an instance of your favorite web browser, and check the terminal window connected to the remote server for a url for Jupyter Notebooks and enter that into your window.
  5. If successful, you should be able to see the files that are stored under your user in the remote server. You should also be able to run notebooks from here, which will execute on the remote clusters.

  FOR MOUNTING ENGRAM SHARES ON VM:<br/>
  **NOTE: Sometimes used memory will accumulate after using this pipeline, especially when the data processing is terminated prematurely or incorrectly. You can free up memory again by rebooting your VM using 'sudo reboot' in the terminal window. When rebooting the VM you will need to mount your engram again. Execute command below to mount engram shares.**
  
  $ sudo /sbin/mount.cifs --verbose -o vers=2.1,user=<YOUR_UNI>,domain=adcu.columbia.edu,uid=$(id -u),forceuid,gid=$(id -g),forcegid,file_mode=0755,dir_mode=0755,rw,noacl //locker-smb.engram.rc.zi.columbia.edu/axel-locker /home/$(id -un)/engram<br/> **Make sure to input your actual UNI**

### align_behavior_and_imaging.py
  * Align collected behavioral data of mouse to neuron calcium imaging data
  * Compared to original data analysis, mine uses floating point zeroes as opposed to integer zeroes.
  
### cnmfe_pipeline.py
  * Main pipeline for motion correction and cnmfE analysis of 1p data. File can be run from the command line, and will take path to data as command line arguments. Can also run multiple files at once, and will ouput a [pickle](https://docs.python.org/3/library/pickle.html) that can later be used for multisession registration across days.
  * Works primarily with .hdf5 files, whose data is stored under an 'images' folder, but this can be modified by altering the 'var_name_hdf5' variable throughout the script. For other file types, edit var_name_hdf5 variable accordingly (default value = 'mov'. This should work with tiffs). 
  * You can run this file either through jupyter notebooks by following the steps outlined above, or you can run it through the terminal by executing a command with the following format: **$ python path/to/cnmfe_pipeline.py path/to/data_directory file1.tif file2.tif file3.tif etc.**
  * See [CaImAn github](https://github.com/flatironinstitute/CaImAn) for details on the package and full paper.

### multisession_registration_prep.py
  * Iterates through every recording under the mouse ID and returns a pickle file that is used for main multisession registration script.

### multisession_registration.py
  * Main multisession registration script. Aligns calcium imaging data across days and returns a list of coordinates with center of mass, contour plot coordinates and bounding box for each component.




### How to install MATLAB on VM
  - **I used MATLAB on the VM in order to run NoRMCorre Motion Correction, but have since switched to CaImAn NoRMCorre which uses Python.**
  - This is a rough walkthrough about how to edit MATLAB scripts while working on a remote server.
  - Follow this guide (https://confluence.columbia.edu/confluence/display/zmbbi/Installing+MATLAB+on+your+Cortex+VM)
  1. Create a Mathworks account. (https://www.mathworks.com/mwaccount/register)<br/>
    * After filling out the info in step 2, you will hit "Create" and will be asked to sign in via your institution. DO NOT CLICK ON THAT BIG BLUE BUTTON !!!<br/>
    * Instead, click on the link below it in the fine print, it will allow you to sign in separately. The idea is that you want to create an account through        Zuckerman, not Columbia.<br/>
    * When filling this next page out, it will ask you for an activation key. Go to [this](https://internal.zi.columbia.edu/sites/default/files/content/zi_matlab_concurrent.txt) link and under where it says "License Activation Key (for association):" you will copy the activation key and paste it into the MathWorks sign up page. Also, save this webpage as a ".lic" file, you'll need it later.<br/>
    * Complete the account registration and verify your email, etc.<br/>
  
  2. Download the X2Go Client and the MATE GUI software on the confluence link from step 1. This step is pretty straightforward, though note that the webpages you download them from do look a bit sketchy. Otherwise, the confluence link does a decent job of explaining how to set it up.<br/>
    * After installing the software, you should now be able to access the VM with a GUI that displays a linux-style GUI from whatever physical machine you're using.
  
  3. Download Chrome to your new linux environment.<br/>
    * In the top left corner of the window, there should be a menu button. Click on it and find the terminal app. <br/>
    * In the terminal, you want to type the following:<br/><br/>
    wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb<br/>
    sudo apt install ./google-chrome-stable_current_amd64.deb<br/><br/>
    * Go to the MathWorks website again and download the MATLAB installer for Linux. Put it in a directory somewhere you're familiar with.<br/>
    * While we're here, you should also move the .lic file we saved earlier. You can do this by opening a terminal window on your local machine and running the following command:<br/><br/>
    scp /path/to/file.lic <your_uni>@smellworld.axel.zi.columbia.edu:~/path/to/directory

  4. Access the MATLAB installer.<br/>
    * Go back to where you put your zipped MATLAB installer in your linux environment, and type in the following command at the linux terminal (and change the year to match whatever current version of MATLAB it is):<br/><br/>
    unzip -X -K matlab_R2021a_glnxa64.zip -d matlab_2021a_installer<br/>
    * If all went well, you should now see a new folder in the same directory called "matlab_2021a_installer" with all of the contents of the zipped folder.<br/>
    * Go back to your terminal and cd into the newly made directory.<br/>
    Run " ./install " and a Matlab installation window should appear.
    
  5. Go through the installation process<br/>
    * The installation will ask you where you want the installation to happen. You can probably do the default location, but I specified my home directory /home/ar4210 for ease of access.<br/>
    * After this window, it might tell you that there isn't enough space on your disk, and the location is /tmp . Email rc@zi.columbia.edu and tell them you need to increase your /tmp partition.<br/>
    * The installation will also ask you to specify the directory where you placed your activation code  (.lic file)
  
  6. Open the MATLAB application<br/>
    * If all went well, you should now be able to go into your terminal and cd into the directory where you said you wanted to place the MATLAB installation, however, you also want to cd into the bin of that directory.<br/>
    * For example, in my case I would run the following command:<br/><br/>
    cd /home/ar4210/bin<br/>
    ./matlab<br/><br/>
    * The first time I tried it I had some weird graphical glitch where parts of the screen went black. If this is the case just log out of the X2Go client, log back in, and try this step again.
    



