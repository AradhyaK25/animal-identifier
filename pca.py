from sklearn.decomposition import PCA
import animal_as_array
        

def pca(X_train,X_test):
    


    #pca = PCA(n_components=None, whiten=True).fit(X_train)
    

    # print pca.explained_variance_ 

    # n_components=0
    # for i in pca.explained_variance_:
    #     if i > 0.04*pca.explained_variance_[0]:
    #         n_components+=1

    n_components = 10 #4%

    pca = PCA(n_components=n_components, whiten=True).fit(X_train)

    eigenanimals = pca.components_.reshape((n_components, 200, 200,3))
    #print n_components
    print "Projecting the input data on the eigenpokemon orthonormal basis"
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    #print X_train_pca
    return X_train_pca,X_test_pca

#print "here"
#pca(animal_as_array.get_animals()['data'])