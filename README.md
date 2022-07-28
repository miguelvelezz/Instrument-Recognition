# Instrument-Recognition: 
# What this is about and how to reproduce my project

## Introduction

For my purposes, instrument recognition is a simplification of a much more complex problem called source separation. Source separation technology can isolate components of a piece of audio. For example, isolating the vocals of a speaker from a clip containing multiple audio sources. With this in mind, I chose to work with computer vision, specifically convolutional neural networks (CNN), to build a classifier determining whether or not an instrument is present in the sample.

This project presents my first step in getting acquainted with the technology used to build source separation systems. We transform the audio files into spectrogram images and use CNNs to classify whether an instrument is present. 

## Dataset: Spotify OpenMIC-2018 

The dataset is published open-source by Spotify and can be found at this link: https://zenodo.org/record/1432913#.YuLWOG_MLy3

Citation: Humphrey, Eric J., Durand, Simon, and McFee, Brian. "OpenMIC-2018: An Open Dataset for Multiple Instrument Recognition." in Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR), 2018.

We use their source audio files as our unstructured data and their target variable from the ".npz" file. 

## Files in repository

* run_scripts.sh
    * Shell script
    * Sequentially runs the python scripts involved in manipulating the source audio files into images and organizing those images into directories of proper train/validation/test splits 
* create_images.py
    * Python script
    * Transforms the audio files in Spotify’s OpenMIC-2018 Dataset into spectrogram images and saves them into a directory created within the script
* reorganize_files.py 
    * Python script
    * Reorganizes the newly constructed images into directories correspond to train, validation, and test sets    
* Instrument Recognition.ipynb
    * Jupyter file
    * Loads my images and leads you through the modeling and analysis
* CondaEnvironment.yml
    * This file contains the specifications of the conda environment used for this project including package versions.

## Instructions

Firstly, make sure to visit the link provided and download Spotify's OpenMIC-2018 dataset. Then make sure you've used "CondaEnvornment.yml" to reproduce the environment of packages that I am working with.

The contents of this repository should be downloaded and placed in a folder with the Spotify dataset. So, for example, you may have a folder on desktop named "Audio Project" containing both the contents of this repo and the folder "openmic-2018" provided by Spotify. 

Now, you run the file "run_scripts.sh" in terminal. This file will run the other python scripts sequentially on its own. The result will be another folder named "Image_Data" containing the subfolders titled "Train", "Validation", and "Test". 

The process of creating and saving the images will likely take several hours. Once this is complete, you can open the jupyter notebook and follow along with the analysis presented therein. Access the jupyter notebook [here](https://github.com/miguelvelezz/Instrument-Recognition/blob/main/Instrument%20Recognition.ipynb).







