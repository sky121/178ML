from scholars import *

def stackPredictions(predictions):
    num_data = len(predictions[0])
    finalPrediction = [0]*num_data
    for j in range(num_data):
        finalPrediction[j] = np.mean(np.array(predictions)[:, j])
    return finalPrediction


def main():
    X = np.genfromtxt('data/X_train.txt', delimiter=',')
    Y = np.genfromtxt('data/Y_train.txt', delimiter=',')
    X_test = np.genfromtxt('data/X_test.txt', delimiter=',')
    X, Y = ml.shuffleData(X, Y)
    #Xtr,Xva,Ytr,Yva = ml.splitData(X,Y,0.75) NOT IMPLEMENTED

    np.random.seed(0)
    
    gradientBoost = GradientBoost(X,Y)
    gradientPrediction = gradientBoost.predict(X_test)[:, 1]
    print(gradientPrediction)
    
    adaBoost = AdaBoost(X,Y)
    adaPrediction = adaBoost.predict(X_test)[:, 1]
    print(adaPrediction)

    gradientBoost2 = GradientBoost2(X,Y)
    gradientPrediction2 = gradientBoost2.predict(X_test)
    print(gradientPrediction2)

    randomForest = RandomForest(X,Y,nFeatures = 50, maxDepth = 15, minLeaf = 4, number_of_learner=25)
    forestPrediction = randomForest.predict(X_test)
    print(forestPrediction)

    randomForest2 = RandomForest2(X,Y)
    forest2Prediction = randomForest2.predict(X_test)
    print(forest2Prediction)

    logistic41features = LogisticRegression().fit(X[:,:41],Y)
    logistic41Prediction = logistic41features.predict(X_test[:,:41])
    print(logistic41Prediction)

    knn28categorical = ml.knn.knnClassify(X[:,41:69],Y)
    knn28Prediction = knn28categorical.predict(X_test[:,41:69])

    gradient38Boost2 = GradientBoost2(X[:,:69], Y)
    gradient38Prediction = gradient38Boost2.predict(X_test[:,:69])

    #finalPrediction = stackPredictions([adaPrediction, forestPrediction, gradientPrediction, forest2Prediction, gradientPrediction2])
    #finalPrediction = 0.2*adaPrediction + 0.05*forestPrediction+ 0.05* forest2Prediction + 0.30*gradientPrediction + 0.35*gradientPrediction2 + 0.05*logistic41Prediction
    #finalPrediction = 0.1*np.array(adaPrediction) + 0.05*np.array(forestPrediction)+ 0.2*np.array(forest2Prediction) + 0.2*np.array(gradientPrediction) + 0.35*np.array(gradientPrediction2) + 0.1*np.array(logistic41Prediction)
    #finalPrediction = 0.15*np.array(adaPrediction)+0.1*np.array(forestPrediction)+ 0.3*np.array(forest2Prediction) + 0.15*np.array(gradientPrediction) + 0.3*np.array(gradientPrediction2)
    
    finalPrediction = 0.1*np.array(adaPrediction) + 0.05*np.array(forestPrediction)+ 0.2*np.array(forest2Prediction) + \
                        0.2*np.array(gradientPrediction) + 0.35*np.array(gradientPrediction2) + 0.02*np.array(logistic41Prediction) + \
                        0.08*np.array(knn28Prediction) + 0*np.array(gradient38Boost2)


    Y_test = np.vstack((np.arange(X_test.shape[0]), finalPrediction)).T
    # Output a file with two columns, a row ID and a confidence in class 1:
    np.savetxt('FinalPredictions.txt', Y_test, '%d, %.2f',header='ID,Predicted', delimiter=',')

main()