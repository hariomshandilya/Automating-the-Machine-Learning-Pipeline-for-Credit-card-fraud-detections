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
Coming to the next cell, this is the most important feature of the library. It allows the training data to be fit and compare to all the algorithms in the library to choose the best one. It displays which model is best and in what evaluation matrix. When the data is imbalance accuracy not always tell you the real story. I checked the precision but AUC, F1 and Kappa score can also be of great help to analyze the models. But this is going to an article amongst itself.

Code: Comparing the model



ADVERTISING



filter_none
brightness_4
# command used for comparing all the models available in the library 
compare_models() 
Output:


Yellow part is the top score for the corresponding model.

Taking a single algorithm performing decently in the comparison and creating a model for the same. The name of the algorithm can be found in the documentation of the pycaret library under creating model

Code: Creating the best model

filter_none
brightness_4
# creating logistic regression model 
ET = create_model('et') 
Code: Displaying the model parameters

filter_none
brightness_4
# displaying the model parameters 
ET 
Output:

Code: Hyperparameter Tuning

filter_none
brightness_4
# hyperparameter tuning for a particular model 
model = tune_model('ET') 
Output:


Code: Saving the model

After hours and hours of training the model and hyper tuning it, the worst thing that can happen to you is that the model disappears as the session time-out occurs. To save you from this nightmare, let me give a trick you will never forget.

filter_none
brightness_4
# saving the model 
save_model(ET, 'ET_saved') 
Code: Loading the model

filter_none
brightness_4
# Loading the saved model 
ET_saved = load_model('ET_saved') 
Output:


Code: Finalizing the Model

A step just before deployment when you merge the train and the validation data and train model on all the data available to you.

filter_none
brightness_4
# finalize a model 
final_rf = finalize_model(rf) 
Deploying the model is deployed on AWS. For the settings required for the same please visit the documentation

filter_none
brightness_4
# Deploy a model 
deploy_model(final_lr, model_name = 'lr_aws', platform = 'aws', authentication = { 'bucket'  : 'pycaret-test' }) 



Recommended Posts:
ML | Credit Card Fraud Detection
Machine Learning for Anomaly Detection
Intrusion Detection System Using Machine Learning Algorithms
How to create a Face Detection Android App using Machine Learning KIT on Firebase
Learning Model Building in Scikit-learn : A Python Machine Learning Library
Artificial intelligence vs Machine Learning vs Deep Learning
How to Start Learning Machine Learning?
Difference Between Artificial Intelligence vs Machine Learning vs Deep Learning
Azure Virtual Machine for Machine Learning
Python | Automating Happy Birthday post on Facebook using Selenium
Real-Time Edge Detection using OpenCV in Python | Canny edge detection method
Python | Corner detection with Harris Corner Detection method using OpenCV
Python | Corner Detection with Shi-Tomasi Corner Detection Method using OpenCV
Object Detection with Detection Transformer (DERT) by Facebook
ML | Unsupervised Face Clustering Pipeline
Prediction using ColumnTransformer, OneHotEncoder and Pipeline
ML | Types of Learning – Supervised Learning
Introduction to Multi-Task Learning(MTL) for Deep Learning
Learning to learn Artificial Intelligence | An overview of Meta-Learning
ML | Reinforcement Learning Algorithm : Python Implementation using Q-learning

amankrsharma3
Check out this Author's contributed articles.
If you like GeeksforGeeks and would like to contribute, you can also write an article using contribute.geeksforgeeks.org or mail your article to contribute@geeksforgeeks.org. See your article appearing on the GeeksforGeeks main page and help other Geeks.

Please Improve this article if you find anything incorrect by clicking on the "Improve Article" button below.


Article Tags : 
Machine Learning
Python
Practice Tags : 
Machine Learning

thumb_up
Be the First to upvote.


 
0

No votes yet.
Feedback/ Suggest ImprovementImprove Article  
Please write to us at contribute@geeksforgeeks.org to report any issue with the above content.
Post navigation
Previous
first_page Wand | path_vertical_line() in Python
Next
last_pagePython – Variable Operations Dictionary update




Writing code in comment? Please use ide.geeksforgeeks.org, generate link and share the link here.


Load Comments

auto

Most popular in Machine Learning
License Plate Recognition with OpenCV and Tesseract OCR
Z score for Outlier Detection - Python
Python - Basics of Pandas using Iris Dataset
Logistic Regression using Statsmodels
Python - Pearson's Chi-Square Test


Most visited in Python
Convert integer to string in Python
How to create an empty DataFrame and append rows & columns to it in Pandas?
Python infinity
Take input from stdin in Python
Program to calculate Electricity Bill


GeeksforGeeks
room
5th Floor, A-118,
Sector-136, Noida, Uttar Pradesh - 201305
email
feedback@geeksforgeeks.org
Company
About Us
Careers
Privacy Policy
Contact Us
Learn
Algorithms
Data Structures
Languages
CS Subjects
Video Tutorials
Practice
Courses
Company-wise
Topic-wise
How to begin?
Contribute
Write an Article
Write Interview Experience
Internships
Videos
@geeksforgeeks , Some rights reserved
We use cookies to ensure you have the best browsing experience on our website. By usin
