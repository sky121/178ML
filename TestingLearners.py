from scholars import *


def stackPredictions(predictions):
    num_data = len(predictions[0])
    finalPrediction = [0]*num_data
    for j in range(num_data):
        finalPrediction[j] = np.mean(np.array(predictions)[:, j])
    return finalPrediction

def convertToFinalPredictions(pred):
    final_pred = []
    for i in pred:
        if i>0.5:
            final_pred.append(1)
        else:
            final_pred.append(0)

    return final_pred

def main():
    X = np.genfromtxt('data/X_train.txt', delimiter=',')
    Y = np.genfromtxt('data/Y_train.txt', delimiter=',')
    X_test = np.genfromtxt('data/X_test.txt', delimiter=',')
    X, Y = ml.shuffleData(X, Y)
    Xtr, Xva, Ytr, Yva = ml.splitData(X, Y, 0.75)

    '''
    nstumps = [20, 50, 100, 200, 500]
    errList = list()
    for s in nstumps:
        adaBoost = AdaBoost(Xtr,Ytr, numStumps=s)
        adaPrediction = adaBoost.predict(Xva)[:, 1]
        errCount = 0
        for pred, va in list(zip(adaPrediction, Yva)):
            if(not ((pred<0.5 and va<0.5) or (pred>0.5 and va>0.5))):
                errCount += 1
        errPercentage = float(errCount)/len(adaPrediction)
        errList.append(errPercentage)
        print("Prediction: ", adaPrediction)
        print("error percentage: ", errPercentage)
    plt.plot(nstumps, errList)
    plt.show()
 
    nstumps = [0.05,0.1,0.25,0.5,1]
    errList = list()
    for s in nstumps:
        adaBoost = AdaBoost(Xtr,Ytr, learning_rate=s)
        adaPrediction = adaBoost.predict(Xva)[:, 1]
        errCount = 0
        for pred, va in list(zip(adaPrediction, Yva)):
            if(not ((pred<0.5 and va<0.5) or (pred>0.5 and va>0.5))):
                errCount += 1
        errPercentage = float(errCount)/len(adaPrediction)
        errList.append(errPercentage)
        print("Prediction: ", adaPrediction)
        print("error percentage: ", errPercentage)
    plt.plot(nstumps, errList)
    plt.show()
 
    nFeatures = [30,50,60,90,107]
    errList = list()
    for f in nFeatures:
        randomForest = RandomForest(Xtr,Ytr,nFeatures = f, maxDepth = 15, minLeaf = 4, number_of_learner=25)
        prediction = randomForest.predict(Xva)
        errCount = 0
        for pred, va in list(zip(prediction, Yva)):
            if(not ((pred<0.5 and va<0.5) or (pred>0.5 and va>0.5))):
                errCount += 1
        errPercentage = float(errCount)/len(prediction)
        errList.append(errPercentage)
        #print("Prediction: ", prediction)
        print("error percentage: ", errPercentage)
    plt.plot(nFeatures, errList)
    plt.show()
  
    nFeatures = [5,10,15,20]
    errList = list()
    for f in nFeatures:
        randomForest = RandomForest(Xtr,Ytr,nFeatures = 50, maxDepth = f, minLeaf = 4, number_of_learner=25)
        prediction = randomForest.predict(Xva)
        errCount = 0
        for pred, va in list(zip(prediction, Yva)):
            if(not ((pred<0.5 and va<0.5) or (pred>0.5 and va>0.5))):
                errCount += 1
        errPercentage = float(errCount)/len(prediction)
        errList.append(errPercentage)
        #print("Prediction: ", prediction)
        print("error percentage: ", errPercentage)
    plt.plot(nFeatures, errList)
    plt.show()
    

    logistic41features = LogisticRegression().fit(Xtr,Ytr)
    pred = logistic41features.predict(Xva)
    print("LOGISTIC ERROR: ", 1-sum(pred==Yva)/len(Yva))
    
    gradientBoost2 = GradientBoost2(Xtr,Ytr)
    gradientPrediction2 = gradientBoost2.predict(Xva)
    gradientPrediction2 = convertToFinalPredictions(gradientPrediction2)
    print("Gradient booting 2",1-sum(gradientPrediction2==Yva)/len(Yva))

    gradientBoost2 = GradientBoost(Xtr,Ytr)
    gradientPrediction2 = gradientBoost2.predict(Xva)[:, 1]
    gradientPrediction2 = convertToFinalPredictions(gradientPrediction2)
    print("GRADIENT ERROR: ",1-sum(gradientPrediction2==Yva)/len(Yva))

    gradientBoost2 = AdaBoost(Xtr,Ytr)
    gradientPrediction2 = gradientBoost2.predict(Xva)[:, 1]
    gradientPrediction2 = convertToFinalPredictions(gradientPrediction2)
    print("AdaBoost ERROR: ",1-sum(gradientPrediction2==Yva)/len(Yva))
    
    print(Xtr[0,38:69])
    knn28categorical = ml.knn.knnClassify(Xtr[:,41:69],Ytr)
    knn28Prediction = knn28categorical.predict(Xva[:,41:69])
    knn28Prediction = convertToFinalPredictions(knn28Prediction)
    print("knn28Prediction ERROR: ",1-sum(knn28Prediction==Yva)/len(Yva))

    '''
    Xtr = np.hstack((Xtr[:,:41],Xtr[:,69:]))
    print(Xtr.shape)
    gradient38Boost2 = GradientBoost2(Xtr, Ytr)
    Xva = np.hstack((Xva[:,:41],Xva[:,69:]))
    print(Xva.shape)
    gradient38Prediction = gradient38Boost2.predict(Xva)
    gradient38Prediction = convertToFinalPredictions(gradient38Prediction)
    print("gradient38Prediction ERROR: ",1-sum(gradient38Prediction==Yva)/len(Yva))

main()
