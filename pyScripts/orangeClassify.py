import Orange
import datetime

def writeConfusionToFile(classifications,data,name):
    with open(name,'w') as writeTo:
        writeTo.write('Classified\Actual,'+','.join(data.domain.class_var.values)+'\n')
        for label in data.domain.class_var.values:
            writeTo.write(label+','+','.join([str(classifications[(str(label2),str(label))])\
                                          for label2 in data.domain.class_var.values])+'\n')

def testSuccess(classifier,test,classifications=None):
    if classifications is None:
        classifications = {}
        for label in test.domain.class_var.values:
            for label2 in test.domain.class_var.values:
				if (str(label),str(label2)) not in classifications:
				    classifications[(str(label),str(label2))] = 0
    correct = 0
    wrong = 0
    for example in test:
        try:
            val = classifier(example)
        except:
            return 'Error'
        classifications[(str(val),str(example['label']))]+=1.0
        if val==example['label']:
            correct+=1.0
        else:
            wrong+=1.0
    return (correct,wrong,classifications)

def getPercentSuccess(result):
    return result[0]/(result[0]+result[1])

def runTest(train,test,learner,totalClassifications=None,writeConfusion=False):
    start = datetime.datetime.today()
    classifier = learner(train)
    end = datetime.datetime.today()
    print 'Start time %s, End time %s, Dif %s'%(str(start),str(end),str(end-start))
    resultTrain=testSuccess(classifier,train,totalClassifications)
    resultTest=testSuccess(classifier,test,totalClassifications)
    if writeConfusion:
        writeConfusionToFile(resultTest[2],test,'Confusion of %s.csv'%learner.name)
    return (resultTrain,resultTest,classifier)

def kFoldCrossValidation(data, learner, returnBest=False, folds=10):
    indices = Orange.data.sample.SubsetIndicesCV(data, folds)
    totalSuccessTrain = 0
    totalSuccessTest = 0
    best = [0,None]
    totalClassifications = {}
    for label in data.domain.class_var.values:
        for label2 in data.domain.class_var.values:
            totalClassifications[(str(label),str(label2))] = 0
    for fold in range(folds):
        train = data.select_ref(indices, fold, negate = 1)
        test  = data.select_ref(indices, fold)
        print "Fold %d" % fold
        resultTrain,resultTest,classifier=runTest(train,test,learner,totalClassifications)
        if resultTest == 'Error':
            print 'Error testing fold, skipping'
            folds=folds-1
        else:
            (correct,wrong,classifications)=resultTrain
            successTrain=correct/(correct+wrong)
            (correct,wrong,classifications)=resultTest
            successTest=correct/(correct+wrong)
            if best[0]<successTest:
                best[1]=classifier
                best[0]=successTest
            totalSuccessTest+=successTest
            totalSuccessTrain+=successTrain
            print 'Success train %f, test %f\n'%(successTrain,successTest)
    writeConfusionToFile(classifications,test,'Confusion of %s.csv'%learner.name)
    averageSuccessTest = totalSuccessTest/folds
    averageSuccessTrain = totalSuccessTrain/folds
    print 'Average success train %f, test %f'%(averageSuccessTrain,averageSuccessTest)
    print '----------------------------------------------------------'
    if returnBest:
        return (best[1],averageSuccess)
    return (averageSuccessTrain,averageSuccessTest)
    
def testTreeLearning(data,test=None,getBest=False, folds=10,writeConfusion=False,**kw):
    print 'Starting Decision Tree learning'
    tree = Orange.classification.tree.TreeLearner(name='Tree',**kw)
    if test is not None:
        (resultTrain, resultTest,classifier)  = runTest(data,test,tree,writeConfusion=writeConfusion)
        return (getPercentSuccess(resultTrain),getPercentSuccess(resultTest))
    return kFoldCrossValidation(data,tree,returnBest=getBest, folds=folds)
    
def testNeuralNetLearning(data,test=None,getBest=False, folds=10,writeConfusion=False,**kw):
    print 'Starting NeuralNet learning'
    net = Orange.classification.neural.NeuralNetworkLearner(name="Net",**kw)
    if test is not None:
        (resultTrain, resultTest,classifier) = runTest(data,test,net,writeConfusion=writeConfusion)
        return (getPercentSuccess(resultTrain),getPercentSuccess(resultTest))
    return kFoldCrossValidation(data,net,returnBest=getBest, folds=folds)
    
def testKNNLearning(data,test=None,getBest=False, folds=10, k=10,writeConfusion=False):
    print 'Starting kNN learning'
    knnLearner = Orange.classification.knn.kNNLearner(name='KNN',k=k)
    if test is not None:
        (resultTrain, resultTest,classifier) = runTest(data,test,knnLearner,writeConfusion=writeConfusion)
        return (getPercentSuccess(resultTrain),getPercentSuccess(resultTest))
    return kFoldCrossValidation(data,knnLearner,returnBest=getBest, folds=folds)
    
def testSVMLearning(data,test=None,getBest=False, folds=10,writeConfusion=False, **kw):
    print 'Starting SVM learning'
    svmLearner = Orange.classification.svm.SVMLearner(name='svm',**kw)
    if test is not None:
        (resultTrain, resultTest,classifier) = runTest(data,test,svmLearner,writeConfusion=writeConfusion)
        return (getPercentSuccess(resultTrain),getPercentSuccess(resultTest))
    return kFoldCrossValidation(data,svmLearner,returnBest=getBest, folds=folds)

def testBoostLearning(data,learner, test=None,getBest=False, folds=10, t=10,writeConfusion=False):
    print 'Starting Boost learning'
    boostLearner = Orange.ensemble.boosting.BoostedLearner(learner,name='Boost',t=t)
    if test is not None:
        (resultTrain, resultTest,classifier)  = runTest(data,test,boostLearner,writeConfusion=writeConfusion)
        return (getPercentSuccess(resultTrain),getPercentSuccess(resultTest))
    return kFoldCrossValidation(data,boostLearner,returnBest=getBest, folds=folds)

def writeToCVS(fileName,values,mode='w'):
    with open(fileName,mode) as writeTo:
        for valList in values:
            writeTo.write(','.join([str(v) for v in valList])+'\n')

def getDataWrapper(maxes,fileName,methodCall):
    successesTrain = ['Train success']
    successesTest = ['Test success']
    for m in maxes[1:]:
         success = methodCall(m)
         successesTrain.append(success[0])
         successesTest.append(success[1])
         writeToCVS(dataName+fileName+'.csv',[maxes,successesTrain,successesTest])

def getTreeData(data,testData=None, fileName = 'StatsTree',pruneDepth = False):
    if pruneDepth:
        maxes = ['Max depth']+[i*5 for i in range(1,5)]+[i*10 for i in range(3,11)]
        getDataWrapper(maxes, fileName, lambda m: testTreeLearning(data,test=testData,max_depth=m,same_majority_pruning=True))
    else:
        maxes = ['Max majority']+[i*0.05 for i in range(4,21)]
        getDataWrapper(maxes, fileName, lambda m: testTreeLearning(data,teste=testData,max_majority=m,same_majority_pruning=True))
   
def getKNNData(data,testData=None, fileName = 'StatsKNN'):    
    maxes = ['k'] + [i*5 for i in range(1,21,2)]
    getDataWrapper(maxes, fileName, lambda m: testKNNLearning(data,test=testData,k=m))

def getSVMData(data,testData=None, kernel = Orange.classification.svm.kernels.RBF, fileName = 'StatsSVM'):
    maxes = ['gamma']+[0.000001,0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
    getDataWrapper(maxes, fileName, lambda m: testSVMLearning(data,test=testData,gamma=m,kernel_type = kernel))

def getNNData(data,testData=None, fileName = 'StatsNN', getEpochs=False):
    if getEpochs:
        maxes = ['Maximum Epochs']+[i*200 for i in range(1,21)]
        getDataWrapper(maxes, fileName, lambda m: testNeuralNetLearning(data,test=testData,max_iter=m))
    else:
        maxes = ['Hidden Layer Nodes']+[5,10,25,50,100,500,1000,5000,10000]
        getDataWrapper(maxes, fileName, lambda m: testNeuralNetLearning(data,test=testData,n_mid=m))

def getBoostingData(data,testData=None, fileName = 'StatsBoost'):
    maxes = ['t']+[i*4 for i in range(1,6)]
    tree = Orange.classification.tree.TreeLearner(name='Tree',max_depth=60,same_majority_pruning=True)
    getDataWrapper(maxes, fileName, lambda m: testBoostLearning(data,tree,test=testData,t=m))

def getLearningCurves():
	tree = Orange.classification.tree.TreeLearner(name='Tree',same_majority_pruning=True)
	successes = [['Number of instances','Tree Train','Tree Test' 'NN Train','NN Test',\
				 'KNN Train','KNN Test','SVM Train','SVM Test', 'Boosting Train', 'Boosting Test']]
	for i in range(1,2):
		newNumInstances = int((i/1)*numInstances)-200
		success = [newNumInstances]
		dataSubset = data.get_items_ref(range(newNumInstances))
		success.append(testTreeLearning(dataSubset,same_majority_pruning=True))
		success.append(testNeuralNetLearning(dataSubset,n_mid=40,normalize=False))
		success.append(testKNNLearning(dataSubset))
		success.append(testSVMLearning(dataSubset,gamma=0.005))
		success.append(testBoostLearning(dataSubset,tree,t=4))
		successes.append(success)
		writeToCVS('learningCurves.csv',successes)

dataName = 'smallEmails'
data = Orange.data.Table("smallEmails.csv")
numInstances = len(data)
data.shuffle()
#test = data.get_items_ref(range(200))
#data = data.get_items_ref(range(200,len(data)))
#testTreeLearning(data,test=test,max_depth=80)
#getTreeData(data,testData=test,fileName = 'StatsTreeNan')
#getTreeData()
#getKNNData(fileName = "statsM")
getNNData(data)
#getNNData(data,testData=test,getEpochs=False,fileName = 'StatsMoreNodes')
#getNNData(getEpochs=True,fileName = 'StatsEpochs')
#getSVMData(kernel = Orange.classification.svm.kernels.RBF,fileName='StatsSVMRBF')
#getSVMData(kernel = Orange.classification.svm.kernels.Polynomial,fileName='StatsSVMPoly')
#getBoostingData(data,testData=test,fileName='StatsBoost',writeConfusion=True)
#tree = Orange.classification.tree.TreeLearner(name='Tree',max_depth=80,same_majority_pruning=True)
#testBoostLearning(data,tree, test=test,getBest=False, t=10,writeConfusion=True)
#getLearningCurves()
