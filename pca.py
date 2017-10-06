from sklearn.decomposition import PCA
import animal_as_array
        

def pca(X_train,X_test):
    n_components = 20


    pca = PCA(n_components=n_components, whiten=True).fit(X_train)
    eigenanimals = pca.components_.reshape((n_components, 100, 100,3))


    print "Projecting the input data on the eigenpokemon orthonormal basis"
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    #print pca.explained_variance_ 
    #print X_train_pca
    return X_train_pca,X_test_pca

#pca(animal_as_array.get_animals()['data'])