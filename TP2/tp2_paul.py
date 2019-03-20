import numpy as np
np.set_printoptions(threshold=np.nan,suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

####Apprentissage supervisé : Feature engineering et classification##########################

##Chargement et préparation des données

#Importation des données
credit_scoring = pd.read_csv('./credit_scoring.csv',sep=';')
nom_cols = credit_scoring.dtypes.index

#Conversion en numpy array
values = credit_scoring.ix[:, 0:13].values
status = credit_scoring.ix[:, 13].values

#Propriétés des données
values.shape #Dimension des données 
hist, bin_edges = np.histogram(status,bins = range(3)) #Nombre de positifs et de négatifs
print(hist)
#Il y a 1216 '0' et 3159 '1'

#Séparation en jdd d'apprentissage et jdd de test
from sklearn.model_selection import train_test_split

values_train, values_test, status_train, status_test = train_test_split(
        values, status, test_size=0.5, random_state=0)

##Apprentissage et évaluation du modèle

def classif(values_train, values_test, status_train, status_test):
    #Apprentissage basé sur un arbre CART
    clf = DecisionTreeClassifier(random_state=1)
    clf.fit(values_train, status_train)
    clf_status_predict = clf.predict(values_test)
    #Apprentissage basé sur les k plus proches voisins.
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(values_train, status_train)
    neigh_status_predict = neigh.predict(values_test)
    print('Accuracy score decision tree :', accuracy_score(status_test,clf_status_predict),\
        '\nAccuracy score k neighbors :', accuracy_score(status_test,neigh_status_predict),\
        '\nRecall score decision tree :', recall_score(status_test,clf_status_predict),\
        '\nRecall score k neighbors :', recall_score(status_test,neigh_status_predict),\
        '\nPrecision score decision tree :', precision_score(status_test,clf_status_predict),\
        '\nPrecision score k neighbors :', precision_score(status_test,neigh_status_predict))
    
print('\n SCORES KNN ET CART')   
classif(values_train, values_test, status_train, status_test)


##Normalisation des variables continues 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(values_train)
values_train_norm = scaler.transform(values_train)
values_test_norm = scaler.transform(values_test)

print('\n\n SCORES KNN ET CART AVEC VALUES NORMALISEES')
classif(values_train_norm, values_test_norm, status_train, status_test)
#Accuracy et précision meilleurs pour le k neighbors. Pas de changements pour 
#le decision tree.

##Création de nouvelles variables caractéristiques par combinaisons linéaires des variables initiales.

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(values_train_norm)


values_train_acp=np.concatenate((values_train_norm,pca.transform(values_train_norm)),axis=1)
values_test_acp=np.concatenate((values_test_norm,pca.transform(values_test_norm)),axis=1)




#Test du code sur les nouvelles données
print('\n\n SCORES KNN ET CART VALUES NORMALISEES ET AXES ACP')
classif(values_train_acp, values_test_acp, status_train, status_test)

#Pas de changements de score notables.

##Sélection de variables

#Importance des variables
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(values_train_norm, status_train)
importances=clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
sorted_idx = np.argsort(importances)[::-1]
features =nom_cols
print(features[sorted_idx])
padding = np.arange(values_train_norm.size/len(values_train_norm)) + 0.5
plt.barh(padding, importances[sorted_idx],xerr=std[sorted_idx], align='center')
plt.yticks(padding, features[sorted_idx])
plt.xlabel("Relative Importance")
plt.title("Variable Importance")
plt.show()


#Nombre de variables à garder
KNN=KNeighborsClassifier(n_neighbors=5)
scores=np.zeros(values_train_norm.shape[1]+1)
for f in np.arange(0, values_train_norm.shape[1]+1):
    X1_f = values_train_norm[:,sorted_idx[:f+1]]
    X2_f = values_test_norm[:,sorted_idx[:f+1]]
    KNN.fit(X1_f,status_train)
    YKNN=KNN.predict(X2_f)
    scores[f]=np.round(accuracy_score(status_test,YKNN),3)
plt.plot(scores)
plt.xlabel("Nombre de Variables")
plt.ylabel("Accuracy")
plt.title("Evolution de l'accuracy en fonction des variables")
plt.show()

#Il serait souhaitable de garder 6 variables.


##Paramétrage des classifieurs
from sklearn.model_selection import GridSearchCV

#Decision tree
# prepare a range of random_states values to test
random_state = [0,1,2,3,4]
# create and fit a ridge regression model, testing each alpha
dt = DecisionTreeClassifier()
grid = GridSearchCV(estimator=dt, param_grid=dict(random_state=random_state),cv=5,scoring="accuracy")
grid.fit(values_train_norm, status_train)
print(grid.best_estimator_.random_state)

#K neighbors
n_neighbors = [2,3,4,5,6]
neigh = KNeighborsClassifier()
grid2 = GridSearchCV(estimator=neigh, param_grid=dict(n_neighbors=n_neighbors),cv=5,scoring="accuracy")
grid2.fit(values_train_norm, status_train)
print(grid2.best_estimator_.n_neighbors)



##Création d'un pipeline 
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

#K neighbors
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', FeatureUnion([('pca', PCA(n_components=3))])),
    ('neigh', KNeighborsClassifier(n_neighbors=5)),
])


#Decision tree
pipeline2 = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', FeatureUnion([('pca', PCA(n_components=3))])),
    ('dt', DecisionTreeClassifier(random_state=2)),
])



##Comparaison algorithmes d'apprentissage 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier    
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import time

values_norm = scaler.fit_transform(values)

clfs = {
'NB': GaussianNB(),
'CART': DecisionTreeClassifier(random_state=2),
'RF': RandomForestClassifier(n_estimators=50),
'KNN': KNeighborsClassifier(n_neighbors=10),
'ID3': DecisionTreeClassifier(criterion = 'entropy',random_state=2),
'MLP': MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(20,10),random_state=1),
'BAGGING': BaggingClassifier(n_estimators=50,random_state=1),
'ADABOOST': AdaBoostClassifier(n_estimators=50)
}


def run_classifiers(clfs,X,Y):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    kf_bis = KFold(n_splits=5, shuffle=True, random_state=0)
    for i in clfs:
        clf = clfs[i]
        print("\n\n======= {0} =======".format(i))
        start_time = time.time()
        cv_acc = cross_val_score(clf, X, Y, cv=kf)
        print('\nTemps d\'exécution d\'un cross val : %s secondes'%(time.time() - start_time))
        cv_auc = cross_val_score(clf, X, Y, cv=kf, scoring='roc_auc')
        cv_prec = cross_val_score(clf, X, Y, cv=kf_bis, scoring='precision')
        print("Accuracy for {0} is: {1:.3f} +/- {2:.3f}".format(i, np.mean(cv_acc), np.std(cv_acc)))
        print("AUC for {0} is: {1:.3f} +/- {2:.3f}".format(i, cv_auc.mean(), cv_auc.std()))
        print("precision for {0} is: {1:.3f} +/- {2:.3f}".format(i, cv_prec.mean(), cv_prec.std()))
run_classifiers(clfs,values_norm,status)

#Les méthodes ensemblistes permettent d'obtenir de meilleurs résultats (Adaboost, RF).
#Cela est dû au fait que ces méthodes utilisent un ensemble de plusieurs classifieurs 
#pour prédire les valeurs de status. Cela résulte en de meilleurs résultats.






####Apprentissage supervisé : Données hétérogènes################################################################

data = pd.read_csv('credit.data', sep='\t',header=None)
print(data.shape)

data = data.ix[:, 0:16].values


#binarisation des targets
data[:,15][data[:,15]=='+']=1
data[:,15][data[:,15]=='-']=0

#conservation uniquement des colonnes de type numerique
data_num=np.hstack((data[:,1:3],data[:,7:8],data[:,13:15],data[:,10:11],data[:,15:16])) 

#valeurs manquantes remplacées par des nan
data_num[data_num=='?']=np.nan

#Valeurs numériques en float
data_num=data_num.astype(dtype='float64')

#suppresion des lignes ayant au moins un nan
mask = np.any(np.isnan(data_num), axis=1)
data_num=data_num[~mask]

#Définition des variables explicatrices et de la variable à exliquer
X=data_num[:,0:6]
Y=data_num[:,6:7]

#taille de l'échantillon
print(X.shape)

#nombre d'exemples positifs et negatifs
plt.hist(Y,bins=2,rwidth=0.5)


#run classifier pour comparer les auc calculés par les différents algorithmes
run_classifiers(clfs,X,Y)

#run classifier sur données normalisées
scaler=StandardScaler()
X_norm = scaler.fit_transform(X)

run_classifiers(clfs,X_norm,Y)


#INTERPRETATION DES RESULTATS



####utilisation de la base originale

##Imputation des valeurs manquantes

var_num=[1, 2, 7, 10, 13, 14]
var_cat=[0, 3, 4, 5, 6, 8, 9, 11, 12]

from sklearn.preprocessing import Imputer
X_cat = np.copy(data[:, var_cat])

#Pour les variables catégorielles
for col_id in range(len(var_cat)):
    unique_val, val_idx = np.unique(X_cat[:, col_id], return_inverse=True)    
    X_cat[:, col_id] = val_idx
 
imp_cat = Imputer(missing_values=0, strategy='most_frequent')
X_cat[:, range(5)] = imp_cat.fit_transform(X_cat[:, range(5)])
print(X_cat.shape)


#Pour les variables numériques
X_num = np.copy(data[:, var_num])
X_num[X_num == '?'] = np.nan
X_num = X_num.astype(float)

imp_num = Imputer(missing_values=np.nan, strategy='mean')
X_num = imp_num.fit_transform(X_num)


#conversion des variables categorielles en variables binaires 
from sklearn.preprocessing import OneHotEncoder
X_cat_bin = OneHotEncoder().fit_transform(X_cat).toarray()

#Il y a donc autant de variables binaires issues de chaque variable catégorielle
#qu'il y avait de modalités dans la variable.



#Construction du jeux de données 
data_bin=np.hstack((X_num,X_cat_bin))

target = data[:,15:16]
target = target.astype(int)


#Run classifier sur données binarisées non normalisées
run_classifiers(clfs,data_bin,target)

#Run classifier sur données binarisées et normalisées
scaler=StandardScaler()
X_num_norm = scaler.fit_transform(X_num)
data_bin_norm=np.hstack((X_num_norm,X_cat_bin))
run_classifiers(clfs,data_bin_norm,target)



'''
########################partie3 ######################################
sms_data=pd.read_table("SMSSpamCollection.data",header=None)
type_sms=sms_data.ix[:, 0:1].values
sms_contenu=sms_data.ix[:, 1:2].values


'''




