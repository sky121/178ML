import mltools as ml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from mltools import kernel
from xgboost import XGBClassifier

import sys
print(sys.path)


class RandomForest:
    def __init__(self, X, Y, nFeatures, maxDepth, minLeaf, number_of_learner):
        (N, D) = X.shape
        self.number_of_learner = number_of_learner
        self.learners = [0]*self.number_of_learner
        for i in range(self.number_of_learner):
            (bstrp_x, bstrp_y) = ml.bootstrapData(X, Y)
            self.learners[i] = ml.dtree.treeClassify(bstrp_x, bstrp_y, nFeatures=nFeatures, maxDepth=maxDepth, minLeaf=minLeaf)
        
        
        
    def predict(self, data):
        predictions = [0]*(self.number_of_learner)
        num_of_data = len(data)
        for i in range(self.number_of_learner):  
            predictions[i] = self.learners[i].predict(data) #prediction = [[data1, data2], [data1, data2], [data1,data2]]
        final_prediction = [0]*num_of_data
        for j in range(num_of_data):
            final_prediction[j] = np.mean(np.array(predictions)[:, j])

        return final_prediction


class AdaBoost:
    def __init__(self, X, Y, numStumps = 100, learning_rate = 0.25):
        self.AdaBoostClassifier = AdaBoostClassifier(n_estimators = numStumps, learning_rate = learning_rate)
        self.AdaBoostClassifier.fit(X, Y)
    
    def predict(self, data):
        return self.AdaBoostClassifier.predict_proba(data)



class GradientBoost:
    def __init__(self, X, Y):
        self.GradientBoostingClassifier = GradientBoostingClassifier()
        self.GradientBoostingClassifier.fit(X, Y)
    
    def predict(self, data):
        return self.GradientBoostingClassifier.predict_proba(data)


class BaggedTree(ml.base.classifier):
    def __init__(self, learners):
        """Constructs a BaggedTree class with a set of learners. """
        self.learners = learners
    
    def predictSoft(self, X):
        """Predicts the probabilities with each bagged learner and average over the results. """
        n_bags = len(self.learners)
        preds = [self.learners[l].predictSoft(X) for l in range(n_bags)]
        return np.mean(preds, axis=0)

class RandomForest2:
    def __init__(self, X, Y, Nbags = 80, maxDepth = 20, nFeatures = 20):
        self.bags = []
        for i in range(Nbags):
            Xi, Yi = ml.bootstrapData(X, Y, X.shape[0])
            tree = ml.dtree.treeClassify(Xi, Yi, maxDepth = maxDepth, nFeatures = nFeatures)
            self.bags.append(tree)
        self.bt = BaggedTree(self.bags)
        self.bt.classes = np.unique(Y)

    def predict(self, data):
        
        x1 = self.bt.predictSoft(data)[:,1]
        return x1
        

class GradientBoost2:
    def __init__(self, X, Y, nEns=100):
        M = X.shape[0]
        self.en = [None]*nEns 
        YHat = np.zeros((M,nEns)) 
        f = np.zeros(Y.shape)
        self.alpha = 0.5
        for l in range(nEns): # this is a lot faster than the bagging loop:
            dJ = 1.*Y - self.sigma(f)
            self.en[l] = ml.dtree.treeRegress(X,dJ, maxDepth=3) # train and save learner
            f -= (self.alpha)*((self.en)[l]).predict(X) 
            
    
    def sigma(self, z): 
        return np.exp(-z)/(1.+np.exp(-z))
    
    def predict(self, data):
        a = np.zeros((data.shape[0],39)) 
        for l in range(39):
            a[:,l] = -self.alpha*self.en[l].predict(data)
        preds = self.sigma(a.sum(axis=1))
        return preds