
# ECEN 649 pattern recognition class project
This code is modified from (https://github.com/sunsided/viola-jones-adaboost/blob/master/viola-jones.ipynb)
## Implement Viola and Jones face detection algorithm

### The algorithm contains four stages,

a.	Haar Feature Selection
As stated in the paper, we have five type rectangular features. The type with two rectangles has horizontal and vertical ones, the type with three rectangles also has horizontal and vertical ones, and the type with four rectangles has one. These five types’ features can represent a human’s upright face. With 24 x 24 pixels.

b.	Creating an Integral Image
To get each feature value efficiently, after load the images and convert them to array, change them into integral image. So each rectangle of any size at any position can be calculated by four addition and subtraction operations. 

c.	Adaboost Training
This part is the main part of Viola and Jones’ algorithm. First train a series of weak classifier, and then combine them into a strong classifier.

d.	Cascading Classifiers
In the first stage, use only two features. In the second stage, train those which classified to be face by the first classifier with ten features. In the following stages, use the images filtered by the previous stage and apply more features on them.

#### The package to import

glob  for deal with the path of the file and data  
os  for input the data  
matplotlib  for plot the picture of the feature and line of system change  
seaborn  for plot the hot pot picture about the prediction rate and false positive,false negative rate   
PIL  for deal with the image, convert the picture to the array  
joblib  for using parallel compute in the code to increase the speed of the code  
sklearn  for using some packeage about machine learning to deal with the algorithm      
warnings for ignoring some warnings when running the project(because i don't write some exception constructure)   

#### How to use
    run violajones.py to finish the algorithm(you can change the round by yourself)(PS. I use FUll PATH of the data, if can't run, please replace it with relative path)    
    run plot.py to construct the feature on the test face  
    run sys_change.py to see the change along the round increasing
