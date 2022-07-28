import os
import time
import json
import shutil
import numpy as np                   
import pandas as pd 

root_directory = 'Image_Data'

train_root = os.path.join(root_directory, 'Train')
validation_root = os.path.join(root_directory, 'Validation')
test_root = os.path.join(root_directory, 'Test')

train_instrument_present_root = os.path.join(train_root, 'Trumpet_Present')
train_instrument_Notpresent_root = os.path.join(train_root, 'Trumpet_Not_Present')

validation_instrument_present_root = os.path.join(validation_root, 'Trumpet_Present')
validation_instrument_Notpresent_root = os.path.join(validation_root, 'Trumpet_Not_Present')

test_instrument_present_root = os.path.join(test_root, 'Trumpet_Present')
test_instrument_Notpresent_root = os.path.join(test_root, 'Trumpet_Not_Present')

if not os.path.exists(train_root):
    os.mkdir(train_root)
    
if not os.path.exists(validation_root):
    os.mkdir(validation_root)

if not os.path.exists(test_root):
    os.mkdir(test_root)
    
if not os.path.exists(train_instrument_present_root):
    os.mkdir(train_instrument_present_root)
if not os.path.exists(train_instrument_Notpresent_root):
    os.mkdir(train_instrument_Notpresent_root)

if not os.path.exists(validation_instrument_present_root):
    os.mkdir(validation_instrument_present_root)
if not os.path.exists(validation_instrument_Notpresent_root):
    os.mkdir(validation_instrument_Notpresent_root)

if not os.path.exists(test_instrument_present_root):
    os.mkdir(test_instrument_present_root)
if not os.path.exists(test_instrument_Notpresent_root):
    os.mkdir(test_instrument_Notpresent_root)

###############################
### Train, Validation, Test ###
###############################

# Train and validation sample keys
train_val_sample_keys = pd.read_csv('openmic-2018/partitions/split01_train.csv',header=None).to_numpy().T[0]
# Shuffling sample keys
np.random.shuffle(train_val_sample_keys)
# Splitting train_val_sample_keys into train and validation - validation is 20% of train
validation_sample_keys,train_sample_keys = np.split(train_val_sample_keys,[int(np.floor(len(train_val_sample_keys)*0.3))])
# Test sample keys are left alone
test_sample_keys =  pd.read_csv('openmic-2018/partitions/split01_test.csv',header=None).to_numpy().T[0]

############################
###   CREATING TARGET    ###
############################

class_map_file = 'openmic-2018/class-map.json'
class_map = json.load(open(class_map_file, 'r'))

npz_file = 'openmic-2018/openmic-2018.npz'
OPENMIC = np.load(npz_file,allow_pickle=True)    
y_mask, sample_key = OPENMIC['Y_mask'], OPENMIC['sample_key']

y_df = pd.DataFrame(y_mask, index=sample_key, columns=class_map.keys()).astype('int32')
target_series = y_df['trumpet']

counter = 1

start_time = time.time()


############################
###     Moving files     ###
############################

# This block of code identifies the directory that the image should belong to and moves it there accordingly. 
# The structure of the folders created is designed to conform to the way "ImageGenerator" in TensorFlow expects 
# your images to saved. 
for filename in os.listdir(root_directory):
    file_path = os.path.join(root_directory, filename)
    
    if os.path.isdir(file_path):
        pass
    else:
        current_sample_key = os.path.splitext(filename)[0] 
        if current_sample_key in train_sample_keys:
            if target_series[current_sample_key] == 1:
                new_path = os.path.join(train_instrument_present_root, filename)
                shutil.move(file_path, new_path)
            else:
                new_path = os.path.join(train_instrument_Notpresent_root, filename)
                shutil.move(file_path, new_path)                
        elif current_sample_key in validation_sample_keys:
            if target_series[current_sample_key] == 1:
                new_path = os.path.join(validation_instrument_present_root, filename)
                shutil.move(file_path, new_path)                 
            else:
                new_path = os.path.join(validation_instrument_Notpresent_root, filename)
                shutil.move(file_path, new_path)              
        elif current_sample_key in test_sample_keys:
            if target_series[current_sample_key] == 1:
                new_path = os.path.join(test_instrument_present_root, filename)
                shutil.move(file_path, new_path)  
            else:
                new_path = os.path.join(test_instrument_Notpresent_root, filename)
                shutil.move(file_path, new_path) 
        else:
            print(f"Sample key {filename} didn't correspond to test or train")