from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pca
import animal_as_array
import numpy as np

dataset = animal_as_array.get_animals()

X = dataset['data'][:100]
Y = dataset['target'][:100]


#print len(X),len(Y)

n_classes = dataset['target_names'].shape[0]

print dataset['target_names']

kf = KFold(len(Y), n_folds=4, shuffle=True)
scores = 0.0

for train_index, test_index in kf:
    X_train = np.array([X[i] for i in train_index])
    X_test = np.array([X[i] for i in test_index])
    Y_train = np.array([Y[i] for i in train_index])
    Y_test = np.array([Y[i] for i in test_index])

    X_train_pca,X_test_pca = pca.pca(X_train,X_test)

    print "Fitting the classifier to the training set"
    param_grid = {
            'kernel': ['rbf', 'linear'],
            'C': [1e3, 5e3, 1e4, 5e4, 1e5],
            'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        }
    clf = GridSearchCV(SVC(class_weight='balanced'), param_grid)
    print "here"
    clf = clf.fit(X_train_pca, Y_train)

    print "Predicting pokemon names on the testing set"
    Y_pred = clf.predict(X_test_pca)

    print classification_report(Y_test, Y_pred, target_names=dataset['target_names'])
    print confusion_matrix(Y_test, Y_pred, labels=range(n_classes))
    scores += clf.score(X_test_pca, Y_test)

