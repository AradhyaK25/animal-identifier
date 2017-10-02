# animal-identifier

This project is an assignment towards the completion of the course BITS F464 Machine Learning. 

This is an image classifier for animals, using an ensemble of classifiers. The identifier works on the same principles as that of facial recognition, drawing heavily from the approaches developed by Sirovich and Kirby in 1987. The crux of the identification lies in dimensionality reduction using Principal Component Analysis (PCA). PCA yields a number of dimensions along which the dataset has most variance, resulting in a number of "eigenanimals". All images can be plotted along these dimensions, to make model fitting less computationally intensive.  

# Dataset
    The dataset used is "Animals on the Web", prepared by Tamara Berg of UNC Chapel Hill 
    (http://tamaraberg.com/animalDataset/index.html), which contains Google Image Search results of various animals, including alligators, dolphins, ants etc. This dataset contains both negative and positive search results, but we shall only use the positive samples for our purpose. 


# Preprocessing
    All images in the given dataset need to be of same dimensions, because their pixels will later be plotted against them in a feature matrix. For this, we shall use Python Image Library (PIL). The images may also be made grayscale if computation is deemed to be too expensive, but that (and also dimension reduction) would be at the expense of loss of information.

