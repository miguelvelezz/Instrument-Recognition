#!/bin/bash

# This file runs "create_images.py" in batches. If all 20,000 audio files are
# transformed into spectrogram images and saved, the python program will use an extraordinary
# amount of RAM. Therefore, in the script "create_images.py", the number of audio files to load is batched. 
# Here we run the script with an argument that specifies what batch to load into the target folder named 
# "Image_Data".
# This allows the program to clear all the objects taking up memory in between batches. 

# The size of all the images in disk should be about 4 GB. 

# The script "reorganize_file.py", creates directories labeled Train, Test, and Validation and moves each
# spectrogram image into its corresponding folder. More information about this file can be found at 
# its header.


max=14
for i in `seq 0 $max`
do
    caffeinate -i python3 create_images.py "$i"
    wait
done


caffeinate -i python3 reorganize_files.py

