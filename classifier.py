from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection, datasets
import pca
import animal_as_array
import numpy as np
import os
from sklearn.externals import joblib
dataset = animal_as_array.get_animals()

X = dataset['data']
Y = dataset['target']
X_test = X[:100]

weight_svm = 0.518
weight_knn = 0.482

probability_threshold = 0.75

#print len(X),len(Y)

n_classes = dataset['target_names'].shape[0]

print dataset['target_names']

# kf = KFold(len(Y), n_folds=4, shuffle=True)
# scores = 0.0
#
# for train_index, test_index in kf:
#     X_train = np.array([X[i] for i in train_index])
#     X_test = np.array([X[i] for i in test_index])
#     Y_train = np.array([Y[i] for i in train_index])
#     Y_test = np.array([Y[i] for i in test_index])
#
#     X_train_pca,X_test_pca = pca.pca(X_train,X_test)
#
#     estimators = []
#     # SVM Classifier
#     print "Fitting the classifier to the training set"
#     param_grid_1 = {
#             'kernel': ['rbf', 'linear'],
#             'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#             'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
#         }
#     clf_1 = GridSearchCV(SVC(class_weight='balanced'), param_grid_1)
#     #print "here"
#     clf_1 = clf_1.fit(X_train_pca, Y_train)
#
#     estimators.append(('SVM',clf_1))
#
#     # K-NN Classifier
#     print "Fitting the classifier to the training set"
#     param_grid_2 = {
#             'n_neighbors': list(xrange(1, 15)),
#     }
#     clf_2 = GridSearchCV(KNeighborsClassifier(), param_grid_2)
#     clf_2 = clf_2.fit(X_train_pca, Y_train)
#
#     estimators.append(('KNN',clf_2))
#
#     print "Predicting pokemon names on the testing set"
#     Y_pred_1 = clf_1.predict(X_test_pca)
#     Y_pred_2 = clf_2.predict(X_test_pca)
#
#     print "report of SVM:"
#     print classification_report(Y_test, Y_pred_1, target_names=dataset['target_names'])
#     #print confusion_matrix(Y_test, Y_pred_1, labels=range(n_classes))
#     #scores += clf.score(X_test_pca, Y_test)
#
#     print "report of KNN:"
#     print classification_report(Y_test, Y_pred_2, target_names=dataset['target_names'])
#     #print confusion_matrix(Y_test, Y_pred_1, labels=range(n_classes))
#     #scores += clf.score(X_test_pca, Y_test)
#
#
#     # ensemble = VotingClassifier(estimators)
#     # results = model_selection.cross_val_score(ensemble, X, Y, cv=kf)
#     # print "ensemble results:",results.mean()


# X_test_dummy = X[:5]
#
# # Train the Classifier and save it to persistent storage.
# X_pca, X_test__dummy_pca = pca.pca(X, X_test_dummy)
#
# # SVM Classifier
# print "Fitting the classifier to the training set"
# param_grid_1 = {
#         'kernel': ['rbf', 'linear'],
#         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#         'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
#     }
# clf_1 = GridSearchCV(SVC(class_weight='balanced', probability=True), param_grid_1)
# #print "here"
# clf_1 = clf_1.fit(X_pca, Y)
# joblib.dump(clf_1, 'svm_trained.pkl')
#
# # KNN Classifier
# print "Fitting the classifier to the training set"
# param_grid_2 = {
#         'n_neighbors': list(xrange(1, 15)),
# }
# clf_2 = GridSearchCV(KNeighborsClassifier(), param_grid_2)
# clf_2 = clf_2.fit(X_pca, Y)
# joblib.dump(clf_2, 'knn_trained.pkl')

def predict(X_test):

    X_pca, X_test_pca = pca.pca(X, X_test)

    print os.getcwd()
    clf_1 = joblib.load('svm_trained.pkl')
    Y_prob_1 = clf_1.predict_proba(X_test_pca)
    # print Y_prob_1

    clf_2 = joblib.load('knn_trained.pkl')
    Y_prob_2 = clf_2.predict_proba(X_test_pca)
    # print Y_prob_2

    Y_prob = [[0 for i in range(len(Y_prob_1[0]))] for i in range(len(X_test))]
    for i in range(len(Y_prob_1)):
        for j in range(len(Y_prob_1[0])):
            Y_prob[i][j] = weight_svm*Y_prob_1[i][j] + weight_knn*Y_prob_2[i][j]

    Y_pred = []
    for ele in Y_prob:
        if ele.index(max(ele)) > probability_threshold:
            Y_pred.append(ele.index(max(ele)))

    frequency_list = [0 for i in range(n_classes)]
    for ele in Y_pred:
        frequency_list[ele] += 1

    print frequency_list

    max_frequency = max(frequency_list)
    predicted_class = []
    predicted_class.append(dataset['target_names'][frequency_list.index(max_frequency)])
    frequency_list[frequency_list.index(max_frequency)] = 0

    for index, ele in enumerate(frequency_list):
        if ele > 0.7 * max_frequency:
            predicted_class.append(dataset['target_names'][index])

    return predicted_class

# predict(X_test)
