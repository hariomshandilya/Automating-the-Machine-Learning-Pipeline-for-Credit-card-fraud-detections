# Automating-the-Machine-Learning-Pipeline-for-Credit-card-fraud-detections
Before going to the code it is requested to work on a Jupyter notebook or ipython notebook. If not installed on your machine you can use Google Collab.This is one of the best and my personal favorite way of working on a python script to work on a Machine Learning problem
Dataset link:
You can download the dataset from this link
If the link is not working please go to this link and login to Kaggle to download the dataset.
Now, I am considering that you have read the previous article without cheating, so let’s proceed further. In this article, I will be using a library known as Pycaret that does all the heavy lifting for me and let me compare the best models side by side with just a few lines of code, which if you remember the first article took us a hell lot of code and all eternity to compare. We also able to do the most cumbersome job in this galaxy other than maintaining 75% attendance, hyperparameter tuning, that takes days and lots of code in just a couple of minutes with a couple of lines of code. It won’t be wrong if you say that this article will be a short and most effective article you will read in a while. So sit back and relax and let the fun begin.

First install the one most important thing that you will need in this article, Pycaret Library. This library is going to save you a ton of money as you know time is money, right.

To install the lib within your Ipython notebook use –
pip install pycaret
Code: Importing the necessary files
filter_none
brightness_4
# importing all necessary libraries 
# linear algebra 
import numpy as np  
# data processing, CSV file I / O (e.g. pd.read_csv) 
import pandas as pd  
Code: Loading the dataset

filter_none
brightness_4
# Load the dataset from the csv file using pandas  
# best way is to mount the drive on colab and   
# copy the path for the csv file  
path ="credit.csv"
data = pd.read_csv(path)  
data.head() 
Code: Knowing the dataset

filter_none
brightness_4
# checking for the imbalance  
len(df[df['Class']== 0]) 
filter_none
brightness_4
len(df[df['Class']== 1]) 
Code: Setting up the pycaret classification

filter_none
brightness_4
# Importing module and initializing setup 
from pycaret.classification import * clf1 = setup(data = df, target = 'Cl
After this, a confirmation will be required to proceed. Press Enter for moving forward with the code.
Check if all the parameters type is correctly identified by the library.
Tell the classifier the percentage of training and validation split is to be taken. I took 80% training data which is quite common in machine learning.
