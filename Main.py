#Load Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
import seaborn as sns
from pylab import savefig
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA



def Load_Data():
    #load Data
    train = pd.read_csv('Data/train.csv')
    test = pd.read_csv('Data/test.csv')

    #Get Columns that are features
    feature_cols = [col for col in train.columns if col not in ['Cover_Type','Id']]

    #Split Data into test, train, cvtrain, and cvtest
    X_train = train[feature_cols]
    X_test = test[feature_cols]
    y = train['Cover_Type']
    test_ids = test['Id']
    
    return X_train, X_test, y, test_ids
    
def Create_CVset(X, y):
    #Create Cross Validation test and training sets    
        
    np.random.seed(1)
    X_cvtrain, X_cvtest, y_cvtrain, y_cvtest = train_test_split(X,y, test_size = 0.3, random_state = 0)
    
    return X_cvtrain, y_cvtrain, X_cvtest, y_cvtest
    
def Accuracy(label, prediction):
    #Returns the multi-class classification accuracy 

    return accuracy_score(label, prediction)

def Explore_Data(X, y):
    #Prints basic statistics about data    
    
    print "Number of training examples: "    
    print np.shape(X)[0]
    print  "Number of features:"
    print np.shape(X)[1]
    print "Number of classification catagories:"    
    print np.nonzero(np.bincount(y))[0].size
    print "Names of catagories:"
    print np.nonzero(np.bincount(y))[0]
    
def Plot_Data(X,y):
    
    #Data
    X = pd.concat([X,y], axis=1)

    Create_Plot(X.ix[:,'Cover_Type'], X.ix[:,'Elevation'])
    Create_Plot(X.ix[:,'Cover_Type'], X.ix[:,'Aspect'])
    Create_Plot(X.ix[:,'Cover_Type'], X.ix[:,'Slope']) 
    Create_Plot(X.ix[:,'Cover_Type'], X.ix[:,'Horizontal_Distance_To_Hydrology'])
    Create_Plot(X.ix[:,'Cover_Type'], X.ix[:,'Hillshade_Noon'])
    Create_Plot(X.ix[:,'Cover_Type'], X.ix[:,'Hillshade_9am'])
    Create_Plot(X.ix[:,'Cover_Type'], X.ix[:,'Hillshade_3pm'])
    Create_Plot(X.ix[:,'Cover_Type'], X.ix[:,'Wilderness_Area1'])
    Create_Plot(X.ix[:,'Cover_Type'], X.ix[:,'Wilderness_Area2'])
    Create_Plot(X.ix[:,'Cover_Type'], X.ix[:,'Wilderness_Area3'])
    Create_Plot(X.ix[:,'Cover_Type'], X.ix[:,'Wilderness_Area4'])
    
    
def Create_Plot(X,y):
    #Creates strip plot of x and y
    xlab = X.name
    ylab = y.name
    xlab = xlab.replace("_"," ")
    ylab = ylab.replace("_"," ")
    figlab = ylab + " vs " + xlab
    filelab =  "Plots/" + figlab.replace(" ","") + ".pdf"
    f, ax = plt.subplots(figsize=(5, 5))
    sns.stripplot(x = X, y = y, jitter = True, size = 5, linewidth = 0.1, ax = ax)
    sns.plt.title(figlab)
    sns.plt.xlabel(xlab)
    sns.plt.ylabel(ylab)
    savefig(filelab)

def Preprocess(X, n):
    #Preprocessing for data
    pca = PCA(n_components = n)
    pca.fit(X)
    X = pca.transform(X)
    print "Explained Variance: "
    print ("{0:.2f}".format(sum(pca.explained_variance_ratio_)*100)) + "%"
    
    return X
    
def fit_predict_model(X,y,XPred, yAns, model):
    #fit a model to to data and make predictions

    if model == "gbm":
        # Setup a GBM classifier
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=10, random_state=0)
        clf.fit(X, y) 
    elif model == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf = clf.fit(X, y)
    elif model == "xrf":
        clf = ExtraTreesClassifier(n_estimators=500, random_state=0)
        clf = clf.fit(X, y)      
    elif model == "vote":
        clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=10, random_state=0)
        clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=10, random_state=1)
        clf3 = RandomForestClassifier(n_estimators=100, random_state=0)
        clf4 = RandomForestClassifier(n_estimators=100, random_state=1)
        clf5 = ExtraTreesClassifier(n_estimators=500, random_state=0)
        clf6 = ExtraTreesClassifier(n_estimators=500, random_state=1)
        
        clf = VotingClassifier(estimators=[('1', clf1), ('2', clf2), ('3', clf3),
                                           ('4', clf4), ('5', clf5), ('6', clf6)
                                           ], voting='hard')
        clf = clf.fit(X, y)
        
    # Make Prediction
    yPred = clf.predict(XPred)
    
    if yAns is not None:
        print "Accuracy for " + model + " :"
        print Accuracy(yAns, yPred)    
    
    else:
        print "Finished training and prediction returned"
        return yPred

def Send_it(ids, pred):
    output = pd.DataFrame({"Id":ids, "Cover_Type":pred})
    output.to_csv("output.csv", index=False)   


def Main():
    
    #Get Data
    X_train, X_test, y_train, test_ids = Load_Data()
    
    #Preprocessing
    #X = pd.concat([X_train, X_test])
    #X = Preprocess(X, 5)
    #X_train, X_test = X[0:X_train.shape[0],:] , X[X_train.shape[0]:,:]    
    
    #Split Data into CV sets
    X_cvtrain, y_cvtrain, X_cvtest, y_cvtest = Create_CVset(X_train, y_train)
        
    #Basic Statistics of Data
    Explore_Data(X_train, y_train)

    #Feature Engineering
    
    
    #View Data    
    Plot_Data(X_train,y_train)
        
    #Find Best model with CV sets
    fit_predict_model(X_cvtrain,y_cvtrain,X_cvtest,y_cvtest, "gbm")
    fit_predict_model(X_cvtrain,y_cvtrain,X_cvtest,y_cvtest, "rf")
    fit_predict_model(X_cvtrain,y_cvtrain,X_cvtest,y_cvtest, "xrf")
    fit_predict_model(X_cvtrain,y_cvtrain,X_cvtest,y_cvtest, "vote")
    
    #Train best model and get predictions on training set
    pred = fit_predict_model(X_train,y_train,X_test,None, "vote")
    
    #Send it! - Create Submission File
    Send_it(test_ids, pred)