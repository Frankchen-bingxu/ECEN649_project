
# ECEN 649 pattern recognition class project
This code is modified from (https://github.com/sunsided/viola-jones-adaboost/blob/master/viola-jones.ipynb)
## Implement Viola and Jones face detection algorithm

### The algorithm contains four stages,

a.	Haar Feature Selection
As stated in the paper, we have five type rectangular features. The type with two rectangles has horizontal and vertical ones, the type with three rectangles also has horizontal and vertical ones, and the type with four rectangles has one. These five types’ features can represent a human’s upright face. With 24 x 24 pixels, we can have 162336 features and with 19 x 19 pixels image, we can have 63960 features in total.

b.	Creating an Integral Image
To get each feature value efficiently, after load the images and convert them to array, change them into integral image. So each rectangle of any size at any position can be calculated by four addition and subtraction operations. 

c.	Adaboost Training
This part is the main part of Viola and Jones’ algorithm. First train a series of weak classifier, and then combine them into a strong classifier.

d.	Cascading Classifiers
In the first stage, use only two features. In the second stage, train those which classified to be face by the first classifier with ten features. In the following stages, use the images filtered by the previous stage and apply more features on them.

#### The package to import

glob  
os  
