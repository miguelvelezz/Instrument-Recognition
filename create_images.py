############################
###     Introduction     ###
############################
# Author: Miguel Velez (4/1)

'''
Before running this file, please ensure that it is contained in the 
same directory as the 'openmic-2018' folder downloaded from Spotify. 

The result of this file is a folder named "Image_Data" with spectrogram images of audio
files in 'openmic-2018' 2018 data set.
'''

############################
###      Importing       ###
############################

import os # For navigating directories
import sys # For using arguments set in terminal
import time # To measure runtime of main code block
import librosa # Library to create spectrogram images
import numpy as np # Mathematical functions like absolute value          
import librosa.display # Required for librosa to display figure
import matplotlib.pyplot as plt # Saving figure        

# openmic-2018 data set in same directory as this notebook
# In "openmic-2018/audio" are many folders with three digit labels "000" to "155". 
# Inside these intermediate file are the .ogg audio samples 

############################
### CREATING DIRECTORIES ###
############################

root_directory = 'openmic-2018/audio'
exit_root = 'Image_Data'

if not os.path.exists(exit_root):
    os.mkdir(exit_root)


############################
###   BATCHING IMAGES    ###
############################

# Helper function to turn the list of directories into a batched list of lists
def split(a, n):
    '''
    Splits a list "a" into n equal parts
    Taken from stack overflow reponse to question:
    https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    '''
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

# The audio files are in directories like 'openmic-2018/audio/152/152972_61440.ogg'
# The three digit subdirectories range from 000-155
# We batch these into groups of eleven or ten so that only images in those folders get loaded when 
# this file is executed 

# The number of bins to divide 000-155 directories into
n_splits = 15
# Creates a list of all the same directories but grouped into sublists
split_directories = list(split(os.listdir(root_directory), n_splits))
# From sys.argv we get the index of the batch that this execution should load
batch_index = int(sys.argv[1])
# Counts the number of intermediate folders 000-155 we have cycled through to keep track of the progress
# of the code
folder_counter = 1

# Start time saves the time stamp of when we enter into the main section of this file
start_time = time.time()


print(f'\nBEGINNING FILE RUN: {batch_index + 1}/{n_splits}...')
# Cycles through all folders labeled 000-155
for foldername in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, foldername)
    # If that folder belongs to the current batch we move on the central operation  
    if foldername in split_directories[batch_index]: 
        
        print(f'\n{foldername} ({folder_counter}/{len(split_directories[batch_index])})\n') 
        folder_counter += 1

        files_counter = 1
        
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):

                # Create image
                wave, sr = librosa.load(os.path.join(folder_path, filename))
                fig = plt.figure(figsize=(6, 3))
                ax = fig.add_subplot(111)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_frame_on(False)
                D = librosa.amplitude_to_db(np.abs(librosa.stft(wave)), ref=np.max)
                librosa.display.specshow(D, y_axis='log',sr=sr)
                # Save image to folder
                out = os.path.join(exit_root, os.path.splitext(filename)[0] + '.png')
                plt.savefig(out,bbox_inches='tight',pad_inches=0)

                print(files_counter/len(os.listdir(folder_path)))
                files_counter += 1

                plt.close('all')
        else:
            print(f'{folder_path} is not a directory')
    else:
        pass        

print('duration (s):  ',(time.time() - start_time),'\n')

