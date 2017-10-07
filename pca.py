from sklearn.decomposition import PCA
import animal_as_array
from PIL import Image


def pca(X_train, X_test):



    #pca = PCA(n_components=None, whiten=True).fit(X_train)


    # print pca.explained_variance_

    # n_components=0
    # for i in pca.explained_variance_:
    #     if i > 0.04*pca.explained_variance_[0]:
    #         n_components+=1

    n_components = 10 #4%

    pca = PCA(n_components=n_components, whiten=True).fit(X_train)

    eigenanimals = pca.components_.reshape((n_components, 300, 300,3))
    #print n_components
    print "Projecting the input data on the eigen animal orthonormal basis"
    X_train_pca = pca.transform(X_train)
    print len(X_train_pca)
    X_test_pca = pca.transform(X_test)

    # reconstruction = pca.inverse_transform(X_train_pca[98])
    # im = Image.fromarray(reconstruction.reshape(300,300,3).astype('uint8'))
    # im.show()
    # im.save('eigen/tiger1.jpg')

    #print X_train_pca
    return X_train_pca, X_test_pca

#print "here"
# data = animal_as_array.get_animals()['data']
# pca(data)
