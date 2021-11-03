#!/usr/bin/env python
# coding: utf-8

'''
Simple Script to automatically run once a week to find and delete memmaps generated from CaImAn Motion Correction / CNMF(E)
Look into cronjobs to see how to automate scripts.

Will search "/home/ar4210/engram/Mouse/Inscopix_Data" and all subdirectories by default. Change Cleanup.base to modify path.

WARNING: Cleanup.backup() is incomplete. 
'''
import os
from distutils.dir_util import copy_tree
from datetime import datetime

class Cleanup:
    base = "/home/ar4210/engram/Mouse/Inscopix_Data"
    
    def __init__(self):
        assert os.path.exists(os.path.join(self.base)), "\033[91mUnable to locate base directory. Please check spelling and location\033[0m"
        print("Root directory found.\033[0m")
        self.log_file = open("/home/ar4210/engram/Mouse/CleanupLog.txt", "a")
        
        self.now = datetime.now()
        self.dt_string = self.now.strftime("%d/%m/%Y %H:%M:%S")
        
    def collect_mmaps(self):
        self.mmap_list = []
        for root, dirs, files in os.walk(os.path.join(self.base)):
            for file in files:
                if f"{root}/{file}".endswith(".mmap"):
                    print(f"{root}/{file}")
                    self.mmap_list.append(f"{root}/{file}")
        
        if len(self.mmap_list) == 0:
            print("No memmaps found.")
            self.log_file.write(f"{self.dt_string} No memmaps found\n")
        else:
            print(f"{len(self.mmap_list)} memmap(s) found\n")
            self.log_file.write(f"{self.dt_string} {len(self.mmap_list)} memmap(s) found.\n")
            for i in self.mmap_list:
                print(i)
                self.log_file.write(f"--------{i}\n")
                
    def delete_mmaps(self):
        if len(self.mmap_list) == 0:
            print("No memory maps collected. Did you run Cleanup.collect_mmaps() ? If yes, there may be no memory maps in the given directory and subdirectories.")
            return
        for i in self.mmap_list:
            assert os.path.exists(i), "cleanup failed! Files not found."
            assert i.endswith(".mmap"), f"cleanup failed! Files not in mmap format: {i}."
            
            print(f"\033[96mBeginning removal of {i}...\033[0m")
            os.remove(i)
            print(f"\033[92mFile {i} successfully removed!\033[0m")
        self.log_file.write(f"{self.dt_string} {len(self.mmap_list)} memmap(s) deleted.\n")
            
    # NOT COMPLETE        
    def backup(self, fromDir = "Test1", toDir = "Test2"):
        assert os.path.exists(os.path.join(self.base, fromDir)), "Origin directory not found."
        assert os.path.exists(os.path.join(self.base, toDir)), "Destination directory not found"
        
        print(f"\033[96mBeginning copying of {os.path.join(self.base, fromDir)} to {os.path.join(self.base, toDir)}...\033[0m")
        copy_tree(os.path.join(self.base, fromDir), os.path.join(self.base, toDir))
        print("Done")

cleanup = Cleanup()
cleanup.collect_mmaps()
cleanup.delete_mmaps()

