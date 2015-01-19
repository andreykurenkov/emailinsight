from sklearn.linear_model import PassiveAggressiveRegressor, PassiveAggressiveClassifier, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from collections import Counter
from nltk.classify import naivebayes
from nltk import compat
from os import listdir
import numpy as np
import scipy as sc
import itertools
import cPickle
import random
import nltk

from reviewDatasetParsing import *
import sys
sys.path.append("..") 
from treestatistics import *

class LenFeatureExtractor:
    def __init__(self):
        self.vectorizer = DictVectorizer(dtype=float, sparse=True)

    def fit(self, X, y=None):
        self.vectorizer.fit([{'textlen':0.0}])
        return self

    def getTextFeatures(self,text):
        return  {'textlen':len(text)}

    def setInstanceNums(self, nums):
        self.instanceNums = nums

    def setInstance(self, num):
        self.atInstance = num

    def transform(self, X, y=None):
        return self.vectorizer.transform([self.getTextFeatures(text) for text in X],y)

class RandomFeatureExtractor:
    def __init__(self):
        self.vectorizer = DictVectorizer(dtype=float, sparse=True)

    def fit(self, X, y=None):
        self.vectorizer.fit([dict([(str(i),0.0) for i in range(10)])])
        return self

    def getTextFeatures(self,text):
        return  dict([(str(i),random.random()) for i in range(10)])

    def setInstanceNums(self, nums):
        self.instanceNums = nums

    def setInstance(self, num):
        self.atInstance = num

    def transform(self, X, y=None):
        return self.vectorizer.transform([self.getTextFeatures(text) for text in X],y)

class FullTextRSTFeatureExtractor:
    def __init__(self,instancenums):
        self.vectorizer = DictVectorizer(dtype=float, sparse=True)
        self.atInstance = 0 
        self.instanceNums = instancenums

    def fit(self, X, y=None):
        self.vectorizer = self.vectorizer.fit([self.getTextFeatures(text) for text in X])
        return self

    def getTextFeatures(self,text):
        features = {}#{'textlen':len(text)}
        rstFile = open('./rstParsed/review%d.brackets'%self.instanceNums[self.atInstance],'r')
        counter = Counter()
        for line in rstFile:
            eduRange, satOrNuc, rangeType  = eval(line)
            counter[satOrNuc] += 1 
            counter[rangeType] += 1 
            counter['lines'] += 1
            counter['maxEDU'] = max(eduRange[1],counter['maxEDU'])
            counter['maxDif'] = max(eduRange[1]-eduRange[0],counter['maxDif'])
        features.update(counter)
        self.atInstance = self.atInstance + 1
        if self.atInstance>=len(self.instanceNums):
            self.atInstance = 0
        return  features

    def setInstanceNums(self, nums):
        self.instanceNums = nums

    def setInstance(self, num):
        self.atInstance = num

    def transform(self, X, y=None):
        return self.vectorizer.transform([self.getTextFeatures(text) for text in X],y)

class LimitedTextRSTFeatureExtractor:
    def __init__(self,instancenums):
        self.vectorizer = DictVectorizer(dtype=float, sparse=True)
        self.atInstance = 0 
        self.instanceNums = instancenums

    def fit(self, X, y=None):
        self.vectorizer = self.vectorizer.fit([self.getTextFeatures(text) for text in X])
        return self

    def getTextFeatures(self,text):
        features = {}#{'textlen':len(text)}
        featuresList = ['conclusion','proportion','result','antithesis','circumstance','otherwise']#,'consequence','contingency']
        rstFile = open('./rstParsed/review%d.brackets'%self.instanceNums[self.atInstance],'r')
        counter = Counter()
        for line in rstFile:
            eduRange, satOrNuc, rangeType  = eval(line)
            if rangeType in featuresList:
                counter[rangeType] += 1 
            #counter['maxEDU'] = max(eduRange[1],counter['maxEDU'])
            #counter['maxDif'] = max(eduRange[1]-eduRange[0],counter['maxDif'])
            #counter['lines'] += 1
        features.update(counter)

        self.atInstance = self.atInstance + 1
        if self.atInstance>=len(self.instanceNums):
            self.atInstance = 0
        return  features

    def setInstanceNums(self, nums):
        self.instanceNums = nums

    def setInstance(self, num):
        self.atInstance = num

    def transform(self, X, y=None):
        return self.vectorizer.transform([self.getTextFeatures(text) for text in X],y)

class FullPickledRSTFeatureExtractor:
    def __init__(self,instancenums):
        self.vectorizer = DictVectorizer(dtype=float, sparse=True)
        self.atInstance = 0 
        self.instanceNums = instancenums

    def fit(self, X, y=None):
        self.vectorizer = self.vectorizer.fit([self.getFeatures(x) for x in X])
        return self
    
    def getFeatures(self,text):
        features = {}#{'textlen':len(text)}
        rstFile = './output_trees/review%d.pickle.gz'%self.instanceNums[self.atInstance]
        tree = getPickledTree(rstFile).tree
        features['size'] = tree_size(tree)
        if features['size']>0:
            features['normdepth'] = tree_depth(tree)/tree_size(tree)
        features['balance'] =abs(tree_balance(tree))
        features.update(relation_proportion(tree)) 
        features.update(parent_relation_proportion(tree)) 
        self.atInstance = self.atInstance + 1
        if self.atInstance>=len(self.instanceNums):
            self.atInstance = 0
        return  features

    def setInstanceNums(self, nums):
        self.instanceNums = nums

    def setInstance(self, num):
        self.atInstance = num

    def transform(self, X, y=None):
        return self.vectorizer.transform([self.getFeatures(x) for x in X],y)

class LimitedPickledRSTFeatureExtractor:
    def __init__(self,instancenums):
        self.vectorizer = DictVectorizer(dtype=float, sparse=True)
        self.atInstance = 0 
        self.instanceNums = instancenums

    def fit(self, X, y=None):
        self.vectorizer = self.vectorizer.fit([self.getFeatures(x) for x in X])
        return self

    def getFeatures(self,text):
        features = {}#{'textlen':len(text)}
        rstFile = './output_trees/review%d.pickle.gz'%self.instanceNums[self.atInstance]
        featureListRelation = ['circumstance','statement']
        featureListRelationP = [('statement', 'condition'),('comment', 'concession'),('cause', 'cause'),
                        ('result', 'explanation'),('contrast', 'contrast'),('contrast', 'comment'),
                        ('summary', 'means'),('manner', 'means'),
                        ('consequence', 'antithesis'),('purpose', 'preference'),
                        ('statement', 'span'),('consequence', 'purpose'),('elaboration', 'statement'),
                        ('explanation', 'result'),('question', 'contrast'),('antithesis', 'reason'),
                        ('purpose', 'disjunction'),('statement', 'elaboration'),('comment', 'means'),
                        ('reason', 'contrast'),('antithesis', 'cause'),('span', 'circumstance'),('summary', 'restatement')]
        tree = getPickledTree(rstFile)
        if tree is not None:
          tree = tree.tree
          d = relation_proportion(tree)
          d2 = parent_relation_proportion(tree)
          for k in featureListRelation:
              if k in d:
                  features[k] = d[k]
          for k in featureListRelationP:
              if k in d2:
                  features[k] = d2[k]
        else:
            print self.instanceNums[self.atInstance]

        self.atInstance = self.atInstance + 1
        if self.atInstance>=len(self.instanceNums):
            self.atInstance = 0
        return  features

    def setInstanceNums(self, nums):
        self.instanceNums = nums

    def setInstance(self, num):
        self.atInstance = num

    def transform(self, X, y=None):
        return self.vectorizer.transform([self.getFeatures(x) for x in X],y)



def runLearner(printStages = True, useSelector = False, discreteHelpfulness = True, useRST = True, useFew = False):
    learner = PassiveAggressiveClassifier() if discreteHelpfulness else PassiveAggressiveRegressor()
    #bestwords = getBestWords(instances,num=1000)
    tfidvec = TfidfVectorizer(sublinear_tf=True,stop_words='english', ngram_range=(1,3), decode_error='replace')
    selector = SelectKBest(chi2, k=50000) if useSelector else None
    encoder = LabelEncoder() if discreteHelpfulness else None
    if discreteHelpfulness:
        classlabels = encoder.fit_transform(labels)
    newData = False

    count = 0
    if useRST:
      print 'Getting RST data'
      nums, texts, ilabels = getPickledRSTSciKitDataLists(True) if newData else getRSTSciKitDataLists(True)

      random = RandomFeatureExtractor()
      lengthBaseline = LenFeatureExtractor()
      fullRST = FullPickledRSTFeatureExtractor(nums)  if newData else FullTextRSTFeatureExtractor(nums)
      limitedRST = LimitedPickledRSTFeatureExtractor(nums)  if newData else LimitedTextRSTFeatureExtractor(nums)
      vectorizer =  FeatureUnion([('extra',limitedRST),('tfid',tfidvec)])

      print 'Fitting random features baseline'
      random.fit(texts)
      print 'Fitting text length baseline'
      lengthBaseline.fit(texts)
      print 'Fitting full RST features'
      fullRST.fit(texts)
      print 'Fitting limited RST features'
      limitedRST.fit(texts)
      print 'Fitting limited RST with tfidvec features'
      vectorizer.fit(texts)
      print 'Fitting tfidvec features'
      tfidvec.fit(texts)

      split = int(0.8*len(ilabels))
      trainData = (texts[:split],ilabels[:split])
      testData = (texts[split:],ilabels[split:])      

      X,y = getAsSciKit(trainData[0],trainData[1],random,encoder,selector)
      learner.fit(X,y)
      X,y = getAsSciKit(trainData[0],trainData[1],random,encoder,selector)
      print 'random features baseline trained on %d instances has accuracy %f'%(len(trainData[0]),learner.score(X,y))

      dummy = DummyClassifier()
      X,y = getAsSciKit(trainData[0],trainData[1],random,encoder,selector)
      dummy.fit(X,y)
      X,y = getAsSciKit(testData[0],testData[1],random,encoder,selector)
      print 'Dummy label distribution baseline trained on %d instances has accuracy %f'%(len(trainData[0]),dummy.score(X,y))

      X,y = getAsSciKit(trainData[0],trainData[1],lengthBaseline,encoder,selector)
      learner.fit(X,y)
      X,y = getAsSciKit(testData[0],testData[1],lengthBaseline,encoder,selector)
      print 'text length baseline trained on %d instances has accuracy %f'%(len(trainData[0]),learner.score(X,y))

      X,y = getAsSciKit(trainData[0],trainData[1],fullRST,encoder,selector)
      learner.fit(X,y)
      X,y = getAsSciKit(testData[0],testData[1],fullRST,encoder,selector)
      print 'Full RST learner trained on %d instances has accuracy %f'%(len(trainData[0]),learner.score(X,y))

      X,y = getAsSciKit(trainData[0],trainData[1],limitedRST,encoder,selector)
      learner.fit(X,y)
      X,y = getAsSciKit(testData[0],testData[1],limitedRST,encoder,selector)
      print 'Limited RST learner trained on %d instances has accuracy %f'%(len(trainData[0]),learner.score(X,y))

      X,y = getAsSciKit(trainData[0],trainData[1],vectorizer,encoder,selector)
      learner.fit(X,y)
      X,y = getAsSciKit(testData[0],testData[1],vectorizer,encoder,selector)
      print 'Limited RST with ngram learner trained on %d instances has accuracy %f'%(len(trainData[0]),learner.score(X,y))

      X,y = getAsSciKit(trainData[0],trainData[1],tfidvec,encoder,selector)
      learner = learner.fit(X,y)
      X,y = getAsSciKit(testData[0],testData[1],tfidvec,encoder,selector)
      print 'ngram learner trained on %d instances has accuracy %f'%(len(trainData[0]),learner.score(X,y))


    else:
      vectorizer = tfidvec
      testData = None
      vocabGotten = False
      instances = ([],[])
      numVocab = 50000
      numTest = 50000
      numTrain = 100000
      maxTrainStages = 20
      for text,label in getSciKitData(stateProgress = False, discreteLabels=discreteHelpfulness):
          if label!='few' or useFew:
            instances[0].append(text)
            instances[1].append(label)
            if not vocabGotten and len(instances[0]) == numVocab:
                if printStages:
                    print 'Fitting vocabulary with %d instances'%numVocab
                vectorizer.fit(instances[0],None)
                if selector is not None:
                    X,y = getSciKitInstance(instances[0],instances[1],vectorizer,encoder,None)
                    selector.fit(X,y)
                vocabGotten = True
                instances = ([],[])
            elif vocabGotten and testData is None and len(instances[0]) == numTest:
                if printStages:
                    print 'Getting test data with %d instances'%numTest
                testData = getSciKitInstance(instances[0],instances[1],vectorizer,encoder,selector)
                instances = ([],[])
            elif vocabGotten and testData is not None and len(instances[0]) == numTrain:
                X,y = getSciKitInstance(instances[0],instances[1],vectorizer,encoder,selector)
                if discreteHelpfulness:
                    learner = learner.partial_fit(X,y, classes = classlabels)
                else:
                    learner = learner.partial_fit(X,y)
                instances = ([],[])
                count = count + 1
                if printStages:
                    print 'Baseline trained on %d instances has accuracy %f'%(count*numTrain,learner.score(testData[0],testData[1]))
            elif count == maxTrainStages:
                break
      print 'Final learner trained on %d instances has accuracy %f'%(maxTrainStages*numTrain,learner.score(testData[0],testData[1]))
if __name__ == '__main__':
    runLearner()
