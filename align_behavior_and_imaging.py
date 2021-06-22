''' 

This script will take raw behavioral data and raw calcium imaging data and align them by frame.
Taken from MATLAB code written by Walter Fischler (wmf2107@columbia.edu) and adapted to Python by Aditya Rao (ar4210@columbia.edu) 

'''

import pandas as pd
import numpy as np
from math import floor, ceil
import os, sys

home_dir = os.getcwd() # --> Users/axellab
data_dir = os.path.join(home_dir, "caiman_data/Example_Data") # --> Users/axellab/caiman_data/Example_Data
print(home_dir, "\n", data_dir)


def calculate_speed(s1, s2, t1, t2):

    '''
    DESCRIPTION:
        Speed Calculation used in main().
    '''

    current_speed = ((s2-s1)*100)/(t2-t1)
    if current_speed > 0:
        return floor(current_speed)
    elif current_speed < 0:
        return ceil(current_speed)
    else:
        return 0

def main():
    '''
    DESCRIPTION:
        Loads neuron spikes data, calcium traces data, and mouse behavior data into pandas DataFrames.
        Aims to align data so that mouse behavior and neuron traces/inferred spikes are conveniently displayed for each
            frame.

    COMMAND LINE ARGS:
        1. .py file
        2. spikes file path
        3. traces file path
        4. behavior file path
        5. file name to save combined spikes and behavior as
        6. file name to save combined traces and behavior as

        python align_behavior_and_imaging.py {2} {3} {4} {5} {6}

    '''

    # spikes data
    # print("Current Directory: " , home_dir)
    # s_df = pd.read_csv(input("Enter Inferred Spikes Data File Path: "), header = None)
    s_df = pd.read_csv(f"{data_dir}/{sys.argv[1]}", header = None)
    
    # traces data
    # print("Current Directory: " , home_dir)
    # traces_df = pd.read_csv(input("Enter Calcium Traces Data File Path: "), header = None)
    traces_df = pd.read_csv(f"{data_dir}/{sys.argv[2]}", header = None)

    # behavior data
    # print("Current Directory: " , home_dir)
    # traces_df = pd.read_csv(input("Enter Inferred Spikes Data File Path: "), header = None)
    behavior_df = pd.read_csv(f"{data_dir}/{sys.argv[3]}", skiprows = 1)



    #Delete garbage rows that show up before the real data
    delete_row = []
    for index, value in enumerate(behavior_df["TTLtotalCount"]):
        if value == 0: break
        else: delete_row.append(index)
    behavior_df.drop(delete_row, axis = 0, inplace = True)
    behavior_df.head()


    # Speed calculation taken and adapted from Walter's MATLAB code
    [nrows, ncols] = behavior_df.shape

    s1 = 0
    s2 = 0
    t1 = 0
    t2 = 0
    last_line_fill = 1
    speed = np.zeros(nrows)
    current_speed = 0


    for i  in range(nrows):
        t2 = behavior_df.iloc[i, 1]
        s2 = behavior_df.iloc[i, 7]
        # change here for 500 to 200
        if (t2 - t1 > 200) and (abs(s2 - s1)< 2000):
            t2 = behavior_df.iloc[i-1, 1];
            s2 = behavior_df.iloc[i-1, 7];
            current_speed = calculate_speed(s1,s2,t1,t2)
            
            t1 = behavior_df.iloc[i,1]
            s1 = behavior_df.iloc[i,7]
            for j in range(last_line_fill, i-1):
                speed[j] = current_speed
            last_line_fill = i
            
        elif (abs(s2-s1)>2000) and (t2-t1 >= 100):
                t2 = behavior_df.iloc[i-1,1]
                s2 = behavior_df.iloc[i-1,7]
                current_speed = calculate_speed(s1,s2,t1,t2)
                for j in range(last_line_fill, i-1):
                    speed[j] = current_speed
                last_line_fill = i
                t1 = behavior_df.iloc[i,1]
                s1 = behavior_df.iloc[i,7]
                
        elif (abs(s2 - s1) > 2000) and (t2-t1 < 100):
            current_speed = speed[last_line_fill]
            # orig for j = last_line_fill+1:i
            for j in range(last_line_fill + 1, i - 1):
                speed[j] = current_speed
            # orig last_line_fill = i+1;
            last_line_fill = i
            t1 = behavior_df.iloc[i,1]
            s1 = behavior_df.iloc[i,7]
            
        elif (i == nrows) and (t2-t1 > 0):
            current_speed = calculate_speed(s1,s2,t1,t2)
            for j in range(last_line_fill, i):
                speed[j] = current_speed


    behavior_df["Speed"] = speed

    behavior_df.drop_duplicates('TTLtotalCount', keep = 'first', inplace = True, ignore_index = True)
    behavior_df.drop(index=behavior_df.index[0], axis=0, inplace=True)

    behavior_df.reset_index(drop = True, inplace = True)
    s_df.reset_index(drop = True, inplace = True)
    traces_df.reset_index(drop = True, inplace = True)

    s_and_behavior = pd.concat([behavior_df, s_df.transpose()], axis = 1, ignore_index = True)
    traces_and_behavior = pd.concat([behavior_df, traces_df.transpose()], axis = 1, ignore_index = True)

    s_and_behavior.to_csv(f"{data_dir}/{sys.argv[4]}", index = False, header = False)
    traces_and_behavior.to_csv(f"{data_dir}/{sys.argv[5]}", index = False, header = False)


if __name__ == "__main__":
    main()



