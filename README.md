# animal-identifier

This project is an assignment towards the completion of the course BITS F464 Machine Learning. 

This is a video classifier for animals, using an ensemble of image classifiers. The identifier works on the same principles as that of facial recognition, drawing heavily from the approaches developed by Sirovich and Kirby in 1987. The crux of the identification lies in dimensionality reduction using Principal Component Analysis (PCA). PCA yields a number of dimensions along which the dataset has most variance, resulting in a number of "eigenanimals". All images can be plotted along these dimensions, to make model fitting less computationally intensive.  


# Problem Statement

In the Pokemon universe, a Pokedex is a device which can be used to identify any Pokemon just by pointing the Pokedex at it, and then also get useful information about the same. Drawing inspiration from it, our idea was to build something to replicate the same functionality for animals, by creating a program to identify animals in a given video.

The way used to achieve this was to create an image classifier for a set of chosen animals, using an ensemble of different classifiers. By running this classifier on frames of a given video, and aggregating the scores, the animals in a video can be identified. 

A major inspiration for the implementation of Support Vector Machines (SVM) classification used was a paper published by L. Sirovich and M. Kirby in 1987, “Low-dimensional procedure for the characterization of human faces”. This paper introduced a computationally inexpensive method of facial recognition, using the Principal Component Analysis (PCA) method of dimensionality reduction. In it, the dimensionality of the training data is first reduced drastically using PCA, and then the testing data is plotted along the dimensions gathered from PCA. This allows only the relevant information to actually influence the classification. Each dimension was called an ‘eigenface’, and each ‘eigenface’ can be thought of as representing a particular feature. The value for a particular ‘eigenface’ indicates the importance of each feature. 


# Dataset
    
The dataset that was originally intended for use was “Animals on the Web”, a dataset created by Tamara Berg of UNC Chapel Hill, but when the classifiers were actually run on the dataset, the precision and recall values were too low to be of use. Upon further inspection, the dataset was not deemed to not have the required quality, along with the presence of a significant number of false positives. Hence, a decision was made to construct a database from scratch, with particular emphasis on the quality of images gathered. The animals chosen were bears, canaries, dolphins, tigers and frogs. 24 high quality images for each animal were gathered from Google Image Search results for the animals.

For preprocessing, all images in the dataset were first reduced to 300x300 dimensions. This was done because each pixel of an image was going to be used as a feature, and all images must have the same number of pixels to allow for classification. The central 300x300 area of the image was taken, since the animal will mostly reside in the centre of the image, and thus allows for least loss of information. 

After this, a 120x300x300x3 matrix was created, containing RGB values of every pixel of every image. Then PCA was applied to this matrix, to determine axes of highest variance. Then the number of dimensions that capture at least 4% of the maximum variance were calculated, which came out to be 10.  Then the original data was fitted to the PCA, to create our training data set. 


# Modelling of the problem

The classification model we decided to use was an ensemble of Support Vector Machines (SVM) and K-Nearest Neighbours (K-NN), because of their success in image recognition applications. The ensemble was chosen to be a weighted average of the scores obtained from SVM and K-NN classification. 

To optimize the models, we ran Grid-Search on the SVM and K-NN models. The Grid-Search for SVM model adjusts parameters within specified values for kernels (‘rbf’ and ‘linear’), C (1000, 5000, 10000, 50000, 100000) and gamma (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1). The Grid-Search for K-NN model adjusts the number of neighbours from 1 to 15 trying to find the best fit for the model.

To determine the weights of the two models, we utilized K-Folds Cross Validation. K-Folds divides the original training data set into ‘k’ partitions, and does ‘k’ iterations over the dataset, taking a different partition as testing data every time, while the rest of the data acts as the training data. K-Folds helps determine the utility of a classification model, while also helping determine the weights to be given to each. 

We ran K-Folds for values of ‘k’ ranging from 3 to 5, and then recorded Average F1-Score values (geometric mean of Average Precision and Average Recall) for the SVM and K-NN models, and then weighted it according to those findings. The recorded values were as given in Figure 1.

Post optimization, to achieve model persistence, we trained the model using the preprocessed data and stored the model as a ‘.pkl’ file which could later be used for prediction. 


# Methodology

This section will describe how our program achieves animal identification from a given input video.

1)  Video Processing Module: The objective of the video processing module is to extract frames from the given video. On average, videos contain 25 frames per second. We decided to extract every 10th frame for reducing computation cost, and also because there is no significant informational difference between adjacent frames. Another reason for doing so is that the number of testing data points ideally shouldn’t be too close to the number of training data points.

2)  Preprocessing extracted frames: The preprocessing module applied to the training data is called for every frame recorded from the video. This module first converts the images into 300x300 dimensions to achieve similarity to the training dataset. These processed images are then converted to an array of 300x300x3 dimensions, containing RGB values. 

3)  Dimensionality Reduction: The array obtained after above preprocessing step, is plotted along the principal components obtained from applying PCA on the training dataset. This transformed array will act as the input to our classification module.

4)  Classification: Initially, we load our SVM and K-NN models from the ‘.pkl’ files generated beforehand. A weighted prediction probability measure is obtained by weighing the prediction probability measures obtained by the SVM and K-NN models using the calculated weights. If the predicted class has a confidence measure of less than 0.75, that sample is discarded because of the possibility of noise, otherwise its identified as the animal for a particular frame. This classification process is repeated for all extracted frames, and the predicted classes are accumulated. The class with the maximum number of samples, as well as all classes with number of samples within 0.7 times the maximum are returned as the predicted animals in the video. These animals are then printed as output to the console.


# Applications

An application that is immediately obvious is identification of animals in the wild in near real time. Zoologists and wildlife enthusiasts could point their mobile phone cameras at any animal, and use the live feed from there as input to the animal identifier to determine which animal it is. More information about the animal could be displayed once the animal is identified, as is done by the Pokedex. On more refinement, the identifier could also determine the exact species of the animal, which could help in discovery of new species in cases where the animal cannot be identified by the classifier.


# Possible improvements:

1) Increasing the number of animal classes to enhance utility

2)Increasing the number of training data points, to improve classification accuracy, and to allow recording of more frames from input video

3) Incorporating an additional classifier on the basis of the sound the animal makes. This classifier can be weighted with the video classifier to improve results.

4) Using more principal components for better reconstruction of original images.


